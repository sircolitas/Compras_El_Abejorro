import os
import json
import datetime
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASA_IGV = 1.18  

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
    raise ValueError("Faltan credenciales en el archivo .env (Revisa Supabase o Gemini)")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

@app.post("/upload")
async def procesar_imagen(file: UploadFile = File(...)):
    print(f"📸 Recibiendo imagen: {file.filename}")
    
    try:
        contenido_imagen = await file.read()
        
        imagen_data = {
            "mime_type": file.content_type,
            "data": contenido_imagen
        }
        
        prompt = """
        Analiza este comprobante de pago de Perú. Extrae los datos y devuélvelos ESTRICTAMENTE en formato JSON. 
        No agregues texto adicional, ni saludos, ni formato markdown (```json). SOLO imprime el JSON puro.
        
        Estructura requerida:
        {
          "ruc_proveedor": "string (11 dígitos. Si no encuentras, usa '00000000000')",
          "razon_social": "string (Nombre de la empresa proveedora, usualmente en la parte superior)",
          "tipo_comprobante": "string (Usa 'Factura' si dice explícitamente Factura y tiene RUC 20, de lo contrario 'Boleta')",
          "fecha_emision": "string (formato YYYY-MM-DD)",
          "monto_total": float (Monto total a pagar del comprobante),
          "lineas": [
            {
              "cantidad": float (La cantidad numérica comprada del producto),
              "descripcion": "string (Nombre o detalle del producto)",
              "precio_total": float (Monto total cobrado por esa cantidad de producto)
            }
          ]
        }
        """
        
        # ==========================================
        # BÚCLE DINÁMICO DE MODELOS (FALLBACK AUTÓNOMO)
        # ==========================================
        print("🔍 Recuperando catálogo de modelos de tu API Key...")
        modelos_disponibles = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Ordenamos la lista para probar primero las versiones "flash"
        modelos_disponibles.sort(key=lambda x: ('flash' in x), reverse=True)
        
        respuesta_ia = None
        modelo_exitoso = ""

        for nombre_modelo in modelos_disponibles:
            try:
                print(f"🔄 Intentando procesar foto con: {nombre_modelo}...")
                modelo = genai.GenerativeModel(nombre_modelo)
                respuesta_ia = modelo.generate_content([prompt, imagen_data])
                modelo_exitoso = nombre_modelo
                print(f"✅ ¡Éxito! El modelo {modelo_exitoso} completó la lectura.")
                break  # Detiene el bucle en cuanto encuentra un modelo que funciona
            except Exception as e:
                # Capturamos el error 404 u otros y pasamos al siguiente silenciosamente
                error_breve = str(e).splitlines()[0]
                print(f"⚠️ {nombre_modelo} rechazado o no disponible: {error_breve}")
                continue
        
        if not respuesta_ia:
            raise Exception("Se probaron todos los modelos disponibles en tu llave, pero ninguno pudo procesar la imagen.")
        
        # ==========================================
        # EXTRACCIÓN Y GUARDADO
        # ==========================================
        texto_respuesta = respuesta_ia.text.strip()
        if texto_respuesta.startswith("```json"):
            texto_respuesta = texto_respuesta[7:]
        if texto_respuesta.endswith("```"):
            texto_respuesta = texto_respuesta[:-3]
            
        texto_respuesta = texto_respuesta.strip()
        datos_ocr = json.loads(texto_respuesta)
        
        tipo_comprobante = datos_ocr.get("tipo_comprobante", "Boleta")
        monto_total = float(datos_ocr.get("monto_total", 0.0))
        
        datos_cabecera = {
            "fecha_emision": datos_ocr.get("fecha_emision", str(datetime.date.today())),
            "ruc_proveedor": str(datos_ocr.get("ruc_proveedor", "00000000000"))[:11],
            "razon_social": str(datos_ocr.get("razon_social", "Proveedor Desconocido")),
            "tipo_comprobante": tipo_comprobante,
            "monto_total": monto_total
        }
        
        respuesta_cabecera = supabase.table("comprobantes").insert(datos_cabecera).execute()
        cabecera_id = respuesta_cabecera.data[0]['id']
        
        lineas_procesadas = []
        
        for linea in datos_ocr.get("lineas", []):
            cantidad = float(linea.get("cantidad", 1.0))
            descripcion = str(linea.get("descripcion", "Sin detalle"))
            precio_total = float(linea.get("precio_total", 0.0))
            
            if tipo_comprobante == "Factura":
                valor_venta = round(precio_total / TASA_IGV, 2)
                igv = round(precio_total - valor_venta, 2)
            else:
                valor_venta = precio_total
                igv = 0.0
                
            lineas_procesadas.append({
                "comprobante_id": cabecera_id,
                "descripcion": descripcion,
                "cantidad": cantidad,
                "valor_venta": valor_venta,
                "igv": igv,
                "precio_total": precio_total
            })

        if lineas_procesadas: 
            supabase.table("lineas_detalle").insert(lineas_procesadas).execute()

        print(f"🚀 ¡Guardado en base de datos! ({len(lineas_procesadas)} productos registrados)")
        return {"status": "success", "message": f"Procesado con IA ({modelo_exitoso})"}
        
    except Exception as e:
        print(f"❌ Error interno: {e}")
        raise HTTPException(status_code=500, detail=str(e))
