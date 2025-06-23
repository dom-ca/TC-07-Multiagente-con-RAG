# SMART-TUTOR: Sistema Multiagente de Tutoría Académica con RAG

**Autores:** Dominic Casares Aguirre (c.2022085016) & Mariana Víquez Monge (c.2022029468)  
**Fecha:** Junio 2025  
**Curso:** TC07 - Sistema Multiagente con RAG

##  Descripción General

SMART-TUTOR es un sistema multiagente inteligente diseñado para proporcionar tutoría académica personalizada en materias STEM (Ciencia, Tecnología, Ingeniería y Matemáticas). El sistema integra técnicas avanzadas de inteligencia artificial generativa, incluyendo **Retrieval-Augmented Generation (RAG)**, para ofrecer una experiencia educativa adaptativa y contextualizada.

### Características Principales

-  **Evaluación automática** del nivel de conocimiento del estudiante
-  **Búsqueda semántica** de recursos educativos mediante RAG
-  **Explicaciones personalizadas** adaptadas al perfil del estudiante
-  **Base de conocimiento extensible** sin reentrenamiento
-  **Arquitectura multiagente** especializada y colaborativa
-  **Procesamiento local** para privacidad de datos

##  Arquitectura del Sistema

El sistema está compuesto por **3 agentes especializados**:

### 1. **Agente Evaluador**
- Analiza la consulta del estudiante
- Determina el nivel de conocimiento (básico, intermedio, avanzado)
- Identifica el tipo de dificultad (conceptual, procedimental, aplicación)
- Proporciona recomendaciones pedagógicas

### 2. **Agente Recuperador RAG**
- Utiliza búsqueda semántica con embeddings
- Recupera recursos educativos relevantes de la base vectorial
- Implementa ranking por similitud semántica
- Filtra contenido según el nivel del estudiante

### 3. **Agente Tutor Coordinador**
- Coordina la información de los otros agentes
- Genera respuestas educativas personalizadas
- Adapta explicaciones al nivel detectado
- Mantiene coherencia y motivación en las respuestas

## Tecnologías Utilizadas

**Modelo Base** | Ollama Llama 3.1 | 8B | Generación de texto y razonamiento |
**Framework LLM** | LangChain | Latest | Integración y manejo de modelos |
**Orquestación** | LangGraph | Latest | Coordinación multiagente |
**Base Vectorial** | FAISS | CPU | Almacenamiento y búsqueda semántica |
**Embeddings** | Ollama Embeddings | Llama 3.1 | Vectorización de texto |
**Lenguaje** | Python | 3.8+ | Implementación principal |

##  Instalación y Configuración

### Prerrequisitos

1. **Python 3.8 o superior**
2. **Ollama instalado y ejecutándose**
   ```bash
   # Instalar Ollama (macOS/Linux)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Descargar modelo Llama 3.1
   ollama pull llama3.1
   ```

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone [url-del-repositorio]
```

### Requerimientos principales

```txt
langchain-community==0.0.38
langgraph==0.0.62
faiss-cpu==1.7.4
ollama>=0.1.0
```

##  Uso del Sistema

### Ejecución Básica

```bash
# Ollama tiene que estar ejecutándose
ollama serve

# En otra terminal, ejecutar el sistema
python main.py
```

### Modos de Uso

1. **Modo Interactivo**: Haz consultas personalizadas
2. **Modo Prueba**: Ejecuta ejemplos predefinidos escribiendo `prueba`

### Materias Disponibles

- `algebra_lineal`: Vectores, matrices, espacios vectoriales, determinantes
- `calculo`: Límites, derivadas, integrales
- `probabilidad`: Conceptos básicos, distribuciones

### Ejemplos de Consultas

```
¿Qué es un vector y cómo se representa?
No entiendo cómo multiplicar matrices, ¿puedes explicármelo?
¿Cuál es la diferencia entre límites y derivadas?
```

##  Configuración Avanzada

### Agregar Contenido Educativo

Modifica la clase `BaseDatosEducativa` para agregar más contenido:

```python
self.contenido_educativo = {
    "nueva_materia": [
        {
            "titulo": "Nuevo Concepto",
            "contenido": "Explicación del concepto...",
            "nivel": "basico",
            "tipo": "concepto"
        }
    ]
}
```

##  Flujo de Procesamiento

![image](https://github.com/user-attachments/assets/7ca1d356-f882-44d1-8664-2379f0f686a3)


##  Casos de Prueba

El sistema incluye casos de prueba predefinidos que demuestran:

- Consultas de nivel básico (conceptos fundamentales)
- Consultas de nivel intermedio (operaciones y procedimientos)
- Consultas de nivel avanzado (teoría y aplicaciones)

##  Características Técnicas

### Escalabilidad
- Arquitectura modular con agentes independientes
- Base vectorial extensible sin límites predefinidos
- Procesamiento asíncrono para múltiples estudiantes

### Personalización
- Evaluación automática de nivel de conocimiento
- Adaptación dinámica de explicaciones
- Historial de aprendizaje personalizado

### Robustez
- Manejo de errores en cada agente
- Fallbacks para garantizar respuestas coherentes
- Validación de estados en cada transición

### Privacidad y Seguridad
- Procesamiento local con Ollama
- Sin envío de datos a servidores externos
- Control total sobre información estudiantil

##  Rendimiento

- **Tiempo de inicialización**: ~10-15 segundos
- **Tiempo de respuesta**: ~5-10 segundos por consulta
- **Memoria RAM**: ~2-4 GB (dependiendo del modelo)
- **Espacio en disco**: ~8 GB (modelo Llama 3.1)

Este sistema requiere recursos computacionales considerables debido al uso de modelos de lenguaje grandes. Se recomienda ejecutar en equipos con al menos 8GB de RAM.
