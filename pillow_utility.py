from PIL import Image, ImageDraw, Image, ImageFont

def draw_borders(pillow_image, bounding, color, image_size, caption='', confidence_score=0):

    # Ustala wielkosc obramowania
    width, height = image_size
    draw = ImageDraw.Draw(pillow_image)
    draw.polygon([
        bounding.normalized_vertices[0].x *
        width, bounding.normalized_vertices[0].y * height,
        bounding.normalized_vertices[1].x *
        width, bounding.normalized_vertices[1].y * height,
        bounding.normalized_vertices[2].x *
        width, bounding.normalized_vertices[2].y * height,
        bounding.normalized_vertices[3].x * width, bounding.normalized_vertices[3].y * height], fill=None, outline=color)

    #TODO: Walidacja czcionki
    font_size = width * height // 22000 if width * height > 400000 else 12

    font = ImageFont.truetype('arial.ttf', 22)

    # Wypisuje nazwe zlokalizowanego obiektu
    draw.text((bounding.normalized_vertices[0].x * width,
               bounding.normalized_vertices[0].y * height), font=font, text=caption, fill=color)

    # Wypisuje pewnosc co do zlokalizowanych obiektow
    draw.text((bounding.normalized_vertices[0].x * width, bounding.normalized_vertices[0].y *
               height + 20), font=font, text='Confidence Score: {0:.2f}'.format(confidence_score), fill=color)

    return pillow_image