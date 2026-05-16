PYTHONPATH=/text_generator/py \
python3 /text_generator/py/generative_poetry/deployment/telegram/run_stressed_gpt_poetry_generation_v3.py \
--mode web \
--host 0.0.0.0 \
--port 7860 \
--log /text_generator/tmp/verslibre_web.log \
--models_dir /text_generator/models \
--tmp_dir /text_generator/tmp
