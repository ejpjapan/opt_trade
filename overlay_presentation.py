

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
import os

ppt_path = Path.home() / 'Dropbox' / 'option_overlay'

prs = Presentation(ppt_path / 'option_overlay.pptx') #option_overlay

TITLE = 0
SUB_TITLE = 1
QUOTE = 2
TITLE_COLUMN1 = 3
TITLE_COLUMN2 = 4
TITLE_COLUMN3 = 5
TITLE_ONLY = 6
CAPTION = 7
BLANK = 8


# slide_layout = prs.slide_layouts[TITLE]


img_path = Path.home() / 'Dropbox' / 'outputDev' / 'fig' / 'heat_map.png'


slide = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY])
# for i in range(0, 8, 1):
#     blank_slide_layout = prs.slide_layouts[i]
#     slide = prs.slides.add_slide(blank_slide_layout)


top = Inches(1.54)
left = Inches(0.28)
height = Inches(3.82)
pic = slide.shapes.add_picture(str(img_path), left, top, height=height)
slide.shapes.title.text = 'Monthly Returns (%)'


prs.save(ppt_path / 'test.pptx')

os.system("open " + str(ppt_path / 'test.pptx'))