

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_PARAGRAPH_ALIGNMENT as PP_ALIGN
import os


def main():
    ppt_path = Path.home() / 'Dropbox' / 'option_overlay'
    fig_path = Path.home() / 'Dropbox' / 'outputDev' / 'fig'
    template_name = 'option_overlay.pptx'
    output_name = 'test.pptx'

    # Assets
    heat_map_path = fig_path / 'heat_map.png'
    cum_perf_path = fig_path / 'cum_perf.png'

    # Layout index
    layout_dict = {'TITLE': 0, 'SUB_TITLE': 1, 'QUOTE': 2, 'TITLE_COLUMN1': 3, 'TITLE_COLUMN2': 4, 'TITLE_COLUMN3': 5,
                   'TITLE_ONLY': 6, 'CAPTION': 7, 'BLANK': 8}

    prs = Presentation(ppt_path / template_name)

    # Title slide
    for shape in prs.slides[0].placeholders:
        print('%d %s' % (shape.placeholder_format.idx, shape.name))
    prs.slides[0].shapes[0].text = 'Generating Income with Index Options'

    # First slide
    slide = prs.slides.add_slide(prs.slide_layouts[layout_dict['TITLE_COLUMN1']])
    for shape in slide.placeholders:
        print('%d %s' % (shape.placeholder_format.idx, shape.name))
    # placeholder = slide.placeholders[1]  # idx key, not position
    slide.shapes.title.text = 'Background'

    paragraph_strs = [
        'Egg, bacon, sausage and spam.',
        'Spam, bacon, sausage and spam.',
        'Spam, egg, spam, spam, bacon and spam.'
    ]
    text_frame = slide.placeholders[1].text_frame
    text_frame.clear()  # remove any existing paragraphs, leaving one empty one

    p = text_frame.paragraphs[0]
    p.text = paragraph_strs[0]
    p.alignment = PP_ALIGN.LEFT

    for para_str in paragraph_strs[1:]:
        p = text_frame.add_paragraph()
        p.text = para_str
        p.alignment = PP_ALIGN.LEFT
        p.level = 1

    # Second slide
    slide = prs.slides.add_slide(prs.slide_layouts[layout_dict['TITLE_ONLY']])
    for shape in slide.placeholders:
        print('%d %s' % (shape.placeholder_format.idx, shape.name))
    placeholder = slide.placeholders[13]  # idx key, not position
    _ = placeholder.insert_picture(str(heat_map_path))
    slide.shapes.title.text = 'Monthly Returns (%)'

    # Third slide
    slide = prs.slides.add_slide(prs.slide_layouts[layout_dict['BLANK']])
    for shape in slide.placeholders:
        print('%d %s' % (shape.placeholder_format.idx, shape.name))
    placeholder = slide.placeholders[13]  # idx key, not position
    _ = placeholder.insert_picture(str(cum_perf_path))
    # slide.shapes.title.text = 'Cumulative Performance'

    # Save and open presentation
    prs.save(ppt_path / output_name)
    os.system("open " + str(ppt_path / output_name))


if __name__ == '__main__':
    main()


# for i in range(0, 8, 1):
#     blank_slide_layout = prs.slide_layouts[i]
#     slide = prs.slides.add_slide(blank_slide_layout)
#
# top = Inches(1.54)
# left = Inches(0.28)
# height = Inches(3.82)
# pic = slide.shapes.add_picture(str(heat_map_path), left, top, height=height)


# for shape in slide.placeholders:
#     print('%d %s' % (shape.placeholder_format.idx, shape.name))