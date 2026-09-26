"""Shared database and OpenRouter clients."""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

DB = os.getenv("NEO4J_DATABASE", "neo4j")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
JEV_MODEL = os.getenv("JEV_MODEL", "typesafe/jev-1.13")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

_neo4j_settings = {
    "NEO4J_URI": os.getenv("NEO4J_URI"),
    "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
}
driver = (
    GraphDatabase.driver(
        _neo4j_settings["NEO4J_URI"],
        auth=(_neo4j_settings["NEO4J_USERNAME"], _neo4j_settings["NEO4J_PASSWORD"]),
    )
    if all(_neo4j_settings.values())
    else None
)
chat = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    if OPENROUTER_API_KEY
    else None
)


def require_driver():
    if driver is None:
        missing = ", ".join(name for name, value in _neo4j_settings.items() if not value)
        raise RuntimeError(f"Set {missing} in .env before connecting to Neo4j.")
    return driver


def query(cypher, **params):
    return require_driver().execute_query(cypher, parameters_=params, database_=DB).records


def embed(text):
    if chat is None:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before generating embeddings.")
    return chat.embeddings.create(model=EMBEDDING_MODEL, input=text).data[0].embedding


if __name__ == "__main__":
    require_driver().verify_connectivity()
    print("neo4j ok, nodes:", query("MATCH (n) RETURN count(n) AS n")[0]["n"])
    print("OpenRouter key:", "configured" if OPENROUTER_API_KEY else "missing")
