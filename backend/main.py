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

class IngresoManual(BaseModel):
    fecha: str
    numero_documento: str
    ruc: str
    razon_social: str
    tipo_comprobante: str
    exonerado_igv: bool
    descripcion: str
    cantidad: float
    valor_venta: float
    igv: float
    total: float

# 4. Motor IA Dinámico
def obtener_modelo_gemini():
    try:
        modelos = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Prioridad 1: Buscar exactamente el modelo 3.6-flash que exige Google actualmente
        for m in modelos:
            if '3.6-flash' in m.lower():
                return genai.GenerativeModel(m)
                
        # Prioridad 2: Si no está el 3.6, buscar cualquier flash que NO sea el 2.5 obsoleto
        for m in modelos:
            if 'flash' in m.lower() and '2.5' not in m.lower():
                return genai.GenerativeModel(m)
                
        # Prioridad 3: El primer modelo que encuentre como último recurso
        return genai.GenerativeModel(modelos[0])
    except Exception as e:
        return genai.GenerativeModel('models/gemini-3.6-flash')

# =====================================================================
# RUTA 1: PROCESAMIENTO MÚLTIPLE CON INTELIGENCIA ARTIFICIAL (IA)
# =====================================================================
@app.post("/upload_multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        files = files[:10]
        
    modelo_ia = obtener_modelo_gemini()
    resultados = []
    
    prompt_financiero = """
    Analiza este documento (puede ser imagen, PDF o texto CSV) de un comprobante de pago.
    Extrae la información en formato JSON estricto sin usar markdown.
    La estructura obligatoria es:
    {
      "cabecera": {
        "fecha_emision": "YYYY-MM-DD",
        "numero_documento": "ej. F001-00123 o vacío",
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
            mime_type = file.content_type
            
            # === CLASIFICADOR DE ARCHIVOS MULTIMODAL ===
            if mime_type.startswith("image/"):
                # Procesar como imagen con Pillow
                contenido_ia = Image.open(io.BytesIO(content))
            elif mime_type == "application/pdf":
                # Procesar como PDF binario para Gemini
                contenido_ia = {"mime_type": "application/pdf", "data": content}
            elif "csv" in mime_type or "text" in mime_type:
                # Procesar como archivo de texto (CSV/TXT)
                texto_doc = content.decode('utf-8', errors='ignore')
                contenido_ia = f"Contenido del documento:\n{texto_doc}"
            else:
                raise Exception(f"Formato no soportado actualmente: {mime_type}")
            
            # Enviar a la IA
            respuesta_ia = modelo_ia.generate_content([prompt_financiero, contenido_ia])
            
            raw_text = respuesta_ia.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3]
                
            datos_json = json.loads(raw_text.strip())
            cabecera = datos_json.get("cabecera", {})
            
            res_cabecera = supabase.table("comprobantes").insert({
                "fecha_emision": cabecera.get("fecha_emision"),
                "numero_documento": cabecera.get("numero_documento", ""),
                "ruc_proveedor": cabecera.get("ruc_proveedor", ""),
                "razon_social": cabecera.get("razon_social", ""),
                "tipo_comprobante": cabecera.get("tipo_comprobante", ""),
                "monto_total": float(cabecera.get("monto_total", 0.0))
            }).execute()
            
            comprobante_id = res_cabecera.data[0]['id']
            
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
        datos_cabecera = {
            "fecha_emision": registro.fecha,
            "numero_documento": registro.numero_documento,
            "ruc_proveedor": registro.ruc,
            "razon_social": registro.razon_social,
            "tipo_comprobante": registro.tipo_comprobante,
            "monto_total": registro.total
        }
        
        res_cabecera = supabase.table("comprobantes").insert(datos_cabecera).execute()
        comprobante_id = res_cabecera.data[0]['id']
        
        supabase.table("lineas_detalle").insert({
            "comprobante_id": comprobante_id,
            "descripcion": registro.descripcion,
            "cantidad": registro.cantidad,
            "valor_venta": registro.valor_venta,
            "igv": registro.igv,
            "precio_total": registro.total
        }).execute()

        return {"mensaje": "Registro manual guardado exitosamente."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en BD: {str(e)}")
