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

# 1. Cargar variables de entorno
load_dotenv()

# 2. Configurar FastAPI y CORS (Permitir conexiones del frontend en Vercel)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Inicializar Supabase y Gemini
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 4. Estructura estricta para el ingreso manual (Validación Pydantic)
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

# 5. Función de Rescate: Bucle dinámico para buscar el mejor modelo de IA disponible
def obtener_modelo_gemini():
    try:
        modelos = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Priorizar siempre los modelos "Flash" por ser más rápidos y económicos
        for m in modelos:
            if 'flash' in m.lower():
                return genai.GenerativeModel(m)
        return genai.GenerativeModel(modelos[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al conectar con Google AI Studio.")

# =====================================================================
# RUTA 1: PROCESAMIENTO MÚLTIPLE CON INTELIGENCIA ARTIFICIAL (IA)
# =====================================================================
@app.post("/upload_multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    # Límite de seguridad de 10 imágenes
    if len(files) > 10:
        files = files[:10]
        
    modelo_ia = obtener_modelo_gemini()
    resultados = []
    
    # Prompt experto para estructuración contable
    prompt_financiero = """
    Analiza esta imagen de un comprobante de pago (boleta, factura o recibo de servicios públicos).
    Extrae la información en formato JSON estricto sin usar markdown (sin ```json).
    Si es un recibo de servicios (luz/agua), trata el consumo mensual como línea de detalle.
    La estructura obligatoria es:
    {
      "cabecera": {
        "fecha": "YYYY-MM-DD",
        "numero_documento": "ej. F001-00123 o vacío",
        "ruc": "solo números o vacío",
        "razon_social": "nombre de la empresa",
        "tipo_comprobante": "Factura, Boleta o Recibo",
        "valor_venta": 0.0,
        "igv": 0.0,
        "total": 0.0
      },
      "lineas_detalle": [
        {
          "cantidad": 0.0,
          "descripcion": "nombre del producto/servicio",
          "precio_unitario": 0.0,
          "subtotal": 0.0
        }
      ]
    }
    """
    
    for file in files:
        try:
            # Leer imagen en memoria y prepararla para Gemini
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            
            # Procesar con IA
            respuesta_ia = modelo_ia.generate_content([prompt_financiero, img])
            
            # Limpiar posible markdown de la respuesta de la IA
            raw_text = respuesta_ia.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3]
                
            datos_json = json.loads(raw_text.strip())
            cabecera = datos_json.get("cabecera", {})
            
            # 1. Guardar Cabecera en Supabase
            res_cabecera = supabase.table("comprobantes").insert({
                "fecha": cabecera.get("fecha"),
                "numero_documento": cabecera.get("numero_documento", ""),
                "ruc": cabecera.get("ruc", ""),
                "razon_social": cabecera.get("razon_social", ""),
                "tipo_comprobante": cabecera.get("tipo_comprobante", ""),
                "exonerado_igv": False, # IA asume False por defecto, ajusta según necesidad
                "valor_venta": float(cabecera.get("valor_venta", 0.0)),
                "igv": float(cabecera.get("igv", 0.0)),
                "total": float(cabecera.get("total", 0.0))
            }).execute()
            
            comprobante_id = res_cabecera.data[0]['id']
            
            # 2. Guardar Líneas de Detalle en Supabase
            lineas = datos_json.get("lineas_detalle", [])
            for linea in lineas:
                supabase.table("lineas_detalle").insert({
                    "comprobante_id": comprobante_id,
                    "cantidad": float(linea.get("cantidad", 1.0)),
                    "descripcion": linea.get("descripcion", ""),
                    "precio_unitario": float(linea.get("precio_unitario", 0.0)),
                    "subtotal": float(linea.get("subtotal", 0.0))
                }).execute()
                
            resultados.append({"archivo": file.filename, "estado": "exitoso"})
            
        except Exception as e:
            resultados.append({"archivo": file.filename, "estado": "error", "detalle": str(e)})
            
    return {"mensaje": "Lote procesado", "resultados": resultados}

# =====================================================================
# RUTA 2: INGRESO MANUAL DIRECTO (Sin IA)
# =====================================================================
@app.post("/upload_manual")
async def upload_manual(registro: IngresoManual):
    try:
        # 1. Guardar Cabecera con datos directos del frontend
        res_cabecera = supabase.table("comprobantes").insert({
            "fecha": registro.fecha,
            "numero_documento": registro.numero_documento,
            "ruc": registro.ruc,
            "razon_social": registro.razon_social,
            "tipo_comprobante": registro.tipo_comprobante,
            "exonerado_igv": registro.exonerado_igv,
            "valor_venta": registro.valor_venta,
            "igv": registro.igv,
            "total": registro.total
        }).execute()
        
        comprobante_id = res_cabecera.data[0]['id']
        
        # 2. Crear una línea de detalle genérica para no romper la estructura relacional
        supabase.table("lineas_detalle").insert({
            "comprobante_id": comprobante_id,
            "cantidad": 1.0,
            "descripcion": "Ingreso Manual / Global",
            "precio_unitario": registro.valor_venta,
            "subtotal": registro.valor_venta
        }).execute()

        return {"mensaje": "Registro manual guardado exitosamente."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en base de datos: {str(e)}")
