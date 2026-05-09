# In[]
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image,plot_results


model = build_sam3_image_model()
model.to("cuda")
processor = Sam3Processor()

# %%
