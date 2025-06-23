"""
TC07 - Sistema Multiagente con RAG
Tutor Académico Personalizado
Implementación con Ollama, LangChain y LangGraph

Agentes:
1. Agente Evaluador: Evalúa el conocimiento del estudiante
2. Agente Recuperador: Busca recursos educativos usando RAG
3. Agente Tutor: Proporciona explicaciones personalizadas y coordina el aprendizaje

Autores: Dominic Casares Aguirre c.2022085016
Mariana Viquez Monge c.2022029468
Fecha: Junio 2025
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime

# Dependencias principales
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class EstudianteState(TypedDict):
    """Estado del estudiante en el sistema"""
    consulta: str
    materia: str
    nivel_conocimiento: str
    evaluacion_resultado: Dict[str, Any]
    recursos_encontrados: List[Dict[str, str]]
    respuesta_final: str
    historial_conversacion: List[str]
    siguiente_accion: str


class BaseDatosEducativa:
    """Simula una base de datos educativa con contenido de ejemplo"""
    
    def __init__(self):
        self.contenido_educativo = {
            "algebra_lineal": [
                {
                    "titulo": "Introducción a Vectores",
                    "contenido": "Un vector es una entidad matemática que tiene magnitud y dirección. En álgebra lineal, los vectores se representan como listas ordenadas de números. Por ejemplo, v = [3, 4] representa un vector en 2D.",
                    "nivel": "basico",
                    "tipo": "concepto"
                },
                {
                    "titulo": "Operaciones con Matrices",
                    "contenido": "Las matrices son arreglos rectangulares de números. La multiplicación de matrices A×B es posible solo si el número de columnas de A es igual al número de filas de B. El resultado es una matriz de dimensión (filas de A) × (columnas de B).",
                    "nivel": "intermedio",
                    "tipo": "operacion"
                },
                {
                    "titulo": "Espacios Vectoriales",
                    "contenido": "Un espacio vectorial es un conjunto de vectores junto con operaciones de suma y multiplicación por escalar que satisfacen ocho axiomas fundamentales: asociatividad, conmutatividad, elemento neutro, elemento inverso, etc.",
                    "nivel": "avanzado",
                    "tipo": "teoria"
                },
                {
                    "titulo": "Determinantes",
                    "contenido": "El determinante de una matriz cuadrada es un número que proporciona información importante sobre la matriz. Para una matriz 2×2, det(A) = ad - bc. Los determinantes nos ayudan a determinar si una matriz es invertible.",
                    "nivel": "intermedio",
                    "tipo": "concepto"
                }
            ],
            "calculo": [
                {
                    "titulo": "Límites",
                    "contenido": "Un límite describe el comportamiento de una función cuando la variable independiente se acerca a un punto específico. Se denota como lim(x→a) f(x) = L, donde L es el valor al que tiende la función.",
                    "nivel": "basico",
                    "tipo": "concepto"
                },
                {
                    "titulo": "Derivadas",
                    "contenido": "La derivada de una función representa la tasa de cambio instantánea. Geométricamente, es la pendiente de la recta tangente a la curva en un punto dado. Se calcula como f'(x) = lim(h→0) [f(x+h) - f(x)]/h.",
                    "nivel": "intermedio",
                    "tipo": "operacion"
                }
            ],
            "probabilidad": [
                {
                    "titulo": "Conceptos Básicos de Probabilidad",
                    "contenido": "La probabilidad mide la incertidumbre de un evento. Varía entre 0 (imposible) y 1 (seguro). La probabilidad de un evento A se calcula como P(A) = número de casos favorables / número de casos totales.",
                    "nivel": "basico",
                    "tipo": "concepto"
                },
                {
                    "titulo": "Distribuciones de Probabilidad",
                    "contenido": "Una distribución de probabilidad describe cómo se distribuyen las probabilidades sobre los valores de una variable aleatoria. Las distribuciones pueden ser discretas (binomial, Poisson) o continuas (normal, exponencial).",
                    "nivel": "avanzado",
                    "tipo": "teoria"
                }
            ]
        }


class SistemaMultiagenteEducativo:
    """Sistema principal que coordina los agentes educativos"""
    
    def __init__(self, modelo_ollama: str = "llama3.1"):
        self.modelo = modelo_ollama
        self.llm = Ollama(model=self.modelo, temperature=0.7)
        self.embeddings = OllamaEmbeddings(model=self.modelo)
        self.base_datos = BaseDatosEducativa()
        self.vector_store = None
        self.inicializar_base_vectorial()
        
    def inicializar_base_vectorial(self):
        """Inicializa la base de datos vectorial con contenido educativo"""
        documentos = []
        
        for materia, contenidos in self.base_datos.contenido_educativo.items():
            for contenido in contenidos:
                doc = Document(
                    page_content=f"Materia: {materia}\nTítulo: {contenido['titulo']}\nNivel: {contenido['nivel']}\nTipo: {contenido['tipo']}\nContenido: {contenido['contenido']}",
                    metadata={
                        "materia": materia,
                        "titulo": contenido['titulo'],
                        "nivel": contenido['nivel'],
                        "tipo": contenido['tipo']
                    }
                )
                documentos.append(doc)
        
        # Dividir documentos en chunks más pequeños
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs_divididos = text_splitter.split_documents(documentos)
        
        # Crear vector store
        self.vector_store = FAISS.from_documents(docs_divididos, self.embeddings)
        print(f" Base vectorial inicializada con {len(docs_divididos)} documentos")

    def agente_evaluador(self, state: EstudianteState) -> EstudianteState:
        """Agente que evalúa el nivel de conocimiento del estudiante"""
        prompt_evaluador = ChatPromptTemplate.from_template("""
        Eres un Agente Evaluador experto en educación. Tu función es evaluar el nivel de conocimiento de un estudiante.

        Consulta del estudiante: {consulta}
        Materia: {materia}

        Analiza la consulta y determina:
        1. Nivel de conocimiento aparente (básico, intermedio, avanzado)
        2. Tipo de dificultad (conceptual, procedimental, aplicación)
        3. Áreas de conocimiento previo necesarias
        4. Recomendaciones para el enfoque de enseñanza

        Proporciona una evaluación estructurada en formato JSON con las claves:
        - nivel_estimado
        - tipo_dificultad
        - conocimientos_previos
        - recomendaciones_ensenanza
        - confianza_evaluacion (0-1)
        """)
        
        try:
            respuesta = self.llm.invoke(
                prompt_evaluador.format(
                    consulta=state["consulta"],
                    materia=state["materia"]
                )
            )
            
            # Procesar respuesta (simplificado para demo)
            evaluacion = {
                "nivel_estimado": "intermedio",
                "tipo_dificultad": "conceptual",
                "conocimientos_previos": ["matemáticas básicas"],
                "recomendaciones_ensenanza": ["usar ejemplos visuales", "partir de conceptos simples"],
                "confianza_evaluacion": 0.8,
                "respuesta_completa": respuesta
            }
            
            state["evaluacion_resultado"] = evaluacion
            state["nivel_conocimiento"] = evaluacion["nivel_estimado"]
            state["siguiente_accion"] = "recuperar_recursos"
            
            print(f" Evaluación completada - Nivel: {evaluacion['nivel_estimado']}")
            
        except Exception as e:
            print(f" Error en agente evaluador: {e}")
            state["evaluacion_resultado"] = {"error": str(e)}
            state["siguiente_accion"] = "recuperar_recursos"
            
        return state

    def agente_recuperador_rag(self, state: EstudianteState) -> EstudianteState:
        """Agente que recupera recursos educativos usando RAG"""
        try:
            # Construir query para búsqueda semántica
            query_busqueda = f"{state['consulta']} {state['materia']} nivel {state['nivel_conocimiento']}"
            
            # Búsqueda semántica en la base vectorial
            documentos_relevantes = self.vector_store.similarity_search(
                query_busqueda, 
                k=3
            )
            
            recursos_encontrados = []
            for doc in documentos_relevantes:
                recurso = {
                    "titulo": doc.metadata.get("titulo", "Sin título"),
                    "contenido": doc.page_content,
                    "nivel": doc.metadata.get("nivel", "no especificado"),
                    "tipo": doc.metadata.get("tipo", "general"),
                    "relevancia": "alta"
                }
                recursos_encontrados.append(recurso)
            
            state["recursos_encontrados"] = recursos_encontrados
            state["siguiente_accion"] = "generar_respuesta"
            
            print(f" Recursos recuperados: {len(recursos_encontrados)} documentos")
            
        except Exception as e:
            print(f" Error en agente recuperador: {e}")
            state["recursos_encontrados"] = []
            state["siguiente_accion"] = "generar_respuesta"
            
        return state

    def agente_tutor_coordinador(self, state: EstudianteState) -> EstudianteState:
        """Agente tutor que coordina y genera la respuesta final personalizada"""
        prompt_tutor = ChatPromptTemplate.from_template("""
        Eres un Tutor Académico Personalizado experto en {materia}. Tu función es proporcionar explicaciones claras y personalizadas.

        INFORMACIÓN DEL ESTUDIANTE:
        - Consulta: {consulta}
        - Nivel estimado: {nivel_conocimiento}
        - Evaluación: {evaluacion}

        RECURSOS DISPONIBLES:
        {recursos}

        INSTRUCCIONES:
        1. Proporciona una explicación clara y adaptada al nivel del estudiante
        2. Usa los recursos encontrados para enriquecer tu respuesta
        3. Incluye ejemplos prácticos cuando sea apropiado
        4. Sugiere pasos siguientes para el aprendizaje
        5. Mantén un tono amigable y motivador

        Genera una respuesta educativa completa y personalizada:
        """)
        
        try:
            # Preparar información de recursos
            recursos_texto = ""
            for i, recurso in enumerate(state["recursos_encontrados"], 1):
                recursos_texto += f"\nRecurso {i}: {recurso['titulo']} ({recurso['nivel']})\n{recurso['contenido'][:200]}...\n"
            
            # Generar respuesta personalizada
            respuesta = self.llm.invoke(
                prompt_tutor.format(
                    materia=state["materia"],
                    consulta=state["consulta"],
                    nivel_conocimiento=state["nivel_conocimiento"],
                    evaluacion=json.dumps(state["evaluacion_resultado"], indent=2),
                    recursos=recursos_texto
                )
            )
            
            state["respuesta_final"] = respuesta
            state["siguiente_accion"] = "finalizar"
            
            # Actualizar historial
            estado_historial = state.get("historial_conversacion", [])
            estado_historial.append(f"Consulta: {state['consulta']}")
            estado_historial.append(f"Respuesta: {respuesta[:100]}...")
            state["historial_conversacion"] = estado_historial
            
            print(" Respuesta tutorial generada exitosamente")
            
        except Exception as e:
            print(f" Error en agente tutor: {e}")
            state["respuesta_final"] = f"Lo siento, hubo un error al generar la respuesta: {e}"
            state["siguiente_accion"] = "finalizar"
            
        return state

    def crear_grafo_multiagente(self):
        """Crea y configura el grafo de estados para el sistema multiagente"""
        # Crear el grafo de estados
        workflow = StateGraph(EstudianteState)
        
        # Agregar nodos (agentes)
        workflow.add_node("evaluador", self.agente_evaluador)
        workflow.add_node("recuperador", self.agente_recuperador_rag)
        workflow.add_node("tutor", self.agente_tutor_coordinador)
        
        # Definir flujo de estados
        workflow.set_entry_point("evaluador")
        
        # Transiciones condicionales
        workflow.add_conditional_edges(
            "evaluador",
            lambda x: x["siguiente_accion"],
            {
                "recuperar_recursos": "recuperador",
                "generar_respuesta": "tutor"
            }
        )
        
        workflow.add_conditional_edges(
            "recuperador", 
            lambda x: x["siguiente_accion"],
            {
                "generar_respuesta": "tutor"
            }
        )
        
        workflow.add_conditional_edges(
            "tutor",
            lambda x: x["siguiente_accion"],
            {
                "finalizar": END
            }
        )
        
        # Compilar el grafo
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        return app

    def procesar_consulta(self, consulta: str, materia: str = "algebra_lineal"):
        """Procesa una consulta del estudiante a través del sistema multiagente"""
        print(f"\n Iniciando consulta: '{consulta}' en materia: {materia}")
        print("=" * 60)
        
        # Estado inicial
        estado_inicial = {
            "consulta": consulta,
            "materia": materia,
            "nivel_conocimiento": "",
            "evaluacion_resultado": {},
            "recursos_encontrados": [],
            "respuesta_final": "",
            "historial_conversacion": [],
            "siguiente_accion": "evaluar"
        }
        
        # Crear y ejecutar el grafo
        app = self.crear_grafo_multiagente()
        
        # Configuración de ejecución
        config = {"configurable": {"thread_id": f"sesion_{datetime.now().isoformat()}"}}
        
        try:
            # Ejecutar el flujo multiagente
            resultado_final = None
            for resultado in app.stream(estado_inicial, config):
                print(f" Procesando estado: {list(resultado.keys())}")
                resultado_final = resultado
                
            return resultado_final
            
        except Exception as e:
            print(f" Error en el procesamiento: {e}")
            return {"error": str(e)}

    def mostrar_resultado(self, resultado):
        """Muestra el resultado final de manera formateada"""
        if "error" in resultado:
            print(f"\n Error: {resultado['error']}")
            return
            
        # Obtener el último estado
        ultimo_estado = list(resultado.values())[-1]
        
        print("\n" + "="*60)
        print(" RESULTADO FINAL DEL SISTEMA MULTIAGENTE")
        print("="*60)
        
        print(f"\n Consulta Original: {ultimo_estado['consulta']}")
        print(f" Materia: {ultimo_estado['materia']}")
        print(f" Nivel Detectado: {ultimo_estado['nivel_conocimiento']}")
        
        if ultimo_estado['evaluacion_resultado']:
            print(f"\n Evaluación:")
            eval_result = ultimo_estado['evaluacion_resultado']
            if 'tipo_dificultad' in eval_result:
                print(f"   - Tipo de dificultad: {eval_result['tipo_dificultad']}")
            if 'confianza_evaluacion' in eval_result:
                print(f"   - Confianza: {eval_result['confianza_evaluacion']:.2f}")
        
        if ultimo_estado['recursos_encontrados']:
            print(f"\n Recursos Encontrados ({len(ultimo_estado['recursos_encontrados'])}):")
            for i, recurso in enumerate(ultimo_estado['recursos_encontrados'], 1):
                print(f"   {i}. {recurso['titulo']} ({recurso['nivel']})")
        
        print(f"\n RESPUESTA DEL TUTOR:")
        print("-" * 40)
        print(ultimo_estado['respuesta_final'])
        print("-" * 40)


def main():
    """Función principal del sistema"""
    print(" SISTEMA MULTIAGENTE - TUTOR ACADÉMICO PERSONALIZADO")
    print("=" * 60)
    print("Implementado con Ollama, LangChain y LangGraph")
    print("Agentes: Evaluador, Recuperador RAG, Tutor Coordinador")
    print()
    
    try:
        # Inicializar sistema
        print("  Inicializando sistema...")
        sistema = SistemaMultiagenteEducativo()
        
        # Casos de prueba
        consultas_prueba = [
            {
                "consulta": "¿Qué es un vector y cómo se representa?",
                "materia": "algebra_lineal"
            },
            {
                "consulta": "No entiendo cómo multiplicar matrices, ¿puedes explicármelo?",
                "materia": "algebra_lineal"
            },
            {
                "consulta": "¿Cuál es la diferencia entre límites y derivadas?",
                "materia": "calculo"
            }
        ]
        
        # Modo interactivo
        print(f"\n MODO INTERACTIVO")
        print("=" * 40)
        print("Ahora puedes hacer tus propias consultas.")
        print("Materias disponibles: algebra_lineal, calculo, probabilidad")
        print("Escribe 'salir' para terminar.\n")
    
        
        while True:
            try:
                consulta = input("Escribe tu consulta ('exit' para terminar, 'prueba' para ejecutar los ejemplos): ").strip()
                if consulta.lower() in ['salir', 'exit', 'quit']:
                    break
                
                if not consulta:
                    continue

                if consulta.lower() == 'prueba':
                    # Procesar consultas de ejemplo
                    print("\n Ejecutando ejemplos de prueba...")
                    for i, caso in enumerate(consultas_prueba, 1):
                        print(f"\n CASO DE PRUEBA {i}")
                        print("=" * 40)
                        
                        resultado = sistema.procesar_consulta(
                            caso["consulta"], 
                            caso["materia"]
                        )
                        
                        sistema.mostrar_resultado(resultado)
                        
                        if i < len(consultas_prueba):
                            input("\n Presiona Enter para continuar con el siguiente caso...")
                else:
                    # Procesar consulta del usuario
                    print("\n Procesando tu consulta...")
                    
                    materia = input("Materia (algebra_lineal/calculo/probabilidad): ").strip()

                    if not materia:
                        materia = "algebra_lineal"
                    
                    print()
                    resultado = sistema.procesar_consulta(consulta, materia)
                    sistema.mostrar_resultado(resultado)
                    print()
                
            except KeyboardInterrupt:
                print("\n\n Saliendo del sistema...")
                break
            except Exception as e:
                print(f"\n Error inesperado: {e}")
                continue
        
    except Exception as e:
        print(f" Error crítico en el sistema: {e}")
    
    print("\n Gracias por usar el Sistema!")


if __name__ == "__main__":
    main()