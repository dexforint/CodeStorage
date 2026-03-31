from inference import get_model

# model_id — берётся со страницы проекта (формат "project-name/version")
model = get_model(
    model_id="human-detection-immug/2", api_key="vhkqYnToBonLsd9aw6qQ"
)  # , api_key="vhkqYnToBonLsd9aw6qQ" HaCaRgxnXByM5tBGjhCZ

# Инференс
# results = model.infer("image.jpg")
# print(results)

# ~/.cache/roboflow/models/
