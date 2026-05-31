import asyncio
import uuid
from services.worker_vision.app.xai import generate_heatmap

async def test():
    with open('tests/test_image.png', 'rb') as f:
        image_bytes = f.read()
    query_id = uuid.uuid4()
    path = await generate_heatmap(image_bytes, query_id)
    print('Heatmap guardado en:', path)

asyncio.run(test())