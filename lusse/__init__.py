import yaml
import logging
from fastapi import FastAPI
from uvicorn.config import LOGGING_CONFIG

app = FastAPI(title="OpenAI-compatible API for ZLAI")
LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


from lusse.routes import *
from lusse.models.manager import register_model_configs


@app.on_event("startup")
async def startup_event():
    """"""
    with open('config.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    app.state.model_managers = register_model_configs(data.get("models"))
