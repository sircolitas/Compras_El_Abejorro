import os
import json
import io
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
from PIL import Image

# 1. Cargar variables
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Inicializar Supabase y Gemini
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Estructura para recibir datos del frontend
class IngresoManual(BaseModel):
    fecha: str
    numero_documento: str
    ruc: str
    razon_social: str
    tipo_comprobante: str
    exonerado_igv: bool
    valor_venta: float
    igv: float
    total: float

# 4. Motor IA Dinámico
def obtener_modelo_gemini():
    try:
        modelos = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in modelos:
            if 'flash' in m.lower():
                return genai.GenerativeModel(m)
        return genai.GenerativeModel(modelos[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error con Google AI Studio.")

# =====================================================================
# RUTA 1: PROCESAMIENTO MÚLTIPLE CON INTELIGENCIA ARTIFICIAL (IA)
# =====================================================================
@app.post("/upload_multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        files = files[:10]
        
    modelo_ia = obtener_modelo_gemini()
    resultados = []
    
    # Prompt adaptado a los nombres exactos de tus columnas
    prompt_financiero = """
    Analiza esta imagen de un comprobante de pago.
    Extrae la información en formato JSON estricto sin usar markdown.
    La estructura obligatoria es:
    {
      "cabecera": {
        "fecha_emision": "YYYY-MM-DD",
        "ruc_proveedor": "solo numeros",
        "razon_social": "nombre de empresa",
        "tipo_comprobante": "Factura, Boleta o Recibo",
        "monto_total": 0.0
      },
      "lineas_detalle": [
        {
          "descripcion": "nombre del producto/servicio",
          "cantidad": 1.0,
          "valor_venta": 0.0,
          "igv": 0.0,
          "precio_total": 0.0
        }
      ]
    }
    """
    
    for file in files:
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            respuesta_ia = modelo_ia.generate_content([prompt_financiero, img])
            
            raw_text = respuesta_ia.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3]
                
            datos_json = json.loads(raw_text.strip())
            cabecera = datos_json.get("cabecera", {})
            
            # Guardado en tabla 'comprobantes' con tus columnas exactas
            res_cabecera = supabase.table("comprobantes").insert({
                "fecha_emision": cabecera.get("fecha_emision"),
                "ruc_proveedor": cabecera.get("ruc_proveedor", ""),
                "razon_social": cabecera.get("razon_social", ""),
                "tipo_comprobante": cabecera.get("tipo_comprobante", ""),
                "monto_total": float(cabecera.get("monto_total", 0.0))
            }).execute()
            
            comprobante_id = res_cabecera.data[0]['id']
            
            # Guardado en tabla 'lineas_detalle' con tus columnas exactas
            lineas = datos_json.get("lineas_detalle", [])
            for linea in lineas:
                supabase.table("lineas_detalle").insert({
                    "comprobante_id": comprobante_id,
                    "descripcion": linea.get("descripcion", ""),
                    "cantidad": float(linea.get("cantidad", 1.0)),
                    "valor_venta": float(linea.get("valor_venta", 0.0)),
                    "igv": float(linea.get("igv", 0.0)),
                    "precio_total": float(linea.get("precio_total", 0.0))
                }).execute()
                
            resultados.append({"archivo": file.filename, "estado": "exitoso"})
            
        except Exception as e:
            resultados.append({"archivo": file.filename, "estado": "error", "detalle": str(e)})
            
    return {"mensaje": "Lote procesado", "resultados": resultados}

# =====================================================================
# RUTA 2: INGRESO MANUAL DIRECTO
# =====================================================================
@app.post("/upload_manual")
async def upload_manual(registro: IngresoManual):
    try:
        # Se mapean los datos visuales a los nombres de tus columnas
        datos_cabecera = {
            "fecha_emision": registro.fecha,
            "ruc_proveedor": registro.ruc,
            "razon_social": registro.razon_social,
            "tipo_comprobante": registro.tipo_comprobante,
            "monto_total": registro.total
        }
        
        res_cabecera = supabase.table("comprobantes").insert(datos_cabecera).execute()
        comprobante_id = res_cabecera.data[0]['id']
        
        # El desglose financiero se guarda en las líneas de detalle
        supabase.table("lineas_detalle").insert({
            "comprobante_id": comprobante_id,
            "descripcion": "Ingreso Manual / Global",
            "cantidad": 1.0,
            "valor_venta": registro.valor_venta,
            "igv": registro.igv,
            "precio_total": registro.total
        }).execute()

        return {"mensaje": "Registro manual guardado exitosamente."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en BD: {str(e)}")
