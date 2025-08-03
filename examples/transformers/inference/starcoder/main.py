import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from generators import GeneratorBase, StarCoder
import json

from util import logger, get_parser

app = FastAPI()
app.add_middleware(
    CORSMiddleware
)
generator: GeneratorBase = ...


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    generated_text: str = generator.generate(inputs, parameters)
    return {
        "generated_text": generated_text.replace(inputs, ""),
        "status": 200
    }


def main():
    global generator
    args = get_parser().parse_args()
    generator = StarCoder(args.pretrained, mirror=args.mirror)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
