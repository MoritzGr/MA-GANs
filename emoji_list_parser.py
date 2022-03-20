import base64
import os

import settings
from PIL import Image
from io import BytesIO


def parse_emoji(lines, category):
    name = lines[14][17:len(lines[14]) - 6]
    name = name.replace(':', '')
    number = lines[0][23:len(lines[0]) - 6]
    if lines[3].startswith('<td class="andr" colspan="11">'):
        return

    save_im(lines[3], name, number, category, 'apple')
    save_im(lines[4], name, number, category, 'google')
    save_im(lines[5], name, number, category, 'facebook')
    save_im(lines[6], name, number, category, 'windows')
    save_im(lines[7], name, number, category, 'twitter')
    save_im(lines[8], name, number, category, 'joypixels')
    save_im(lines[9], name, number, category, 'samsung')


def save_im(line, name, number, category, manufacturer):
    # print(number, name, category, manufacturer)
    try:
        im = Image.open(BytesIO(base64.b64decode(line.split('"')[7].split(',')[1])))

        filepath = settings.EMOJI_IMAGES_PATH + '/by_manufacturer/' + manufacturer + '/'
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        im.save(filepath + number + '_' + name + '.png')

        filepath = settings.EMOJI_IMAGES_PATH + '/by_category/' + category + '/'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        im.save(filepath + number + '_' + name + '_' + manufacturer + '.png')

        if category.__contains__('face'):
            filepath = settings.EMOJI_IMAGES_PATH + '/by_category_faces_only/' + category + '/'
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            im.save(filepath + number + '_' + name + '_' + manufacturer + '.png')

            filepath = settings.EMOJI_IMAGES_PATH + '/by_manufacturer_faces_only/' + manufacturer + '/'
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            im.save(filepath + number + '_' + name + '_' + '.png')
    except:
        print("error: skipped " + number + '_' + name)
        pass


def parse_emoji_list():
    with open(settings.EMOJI_LIST_PATH, "r", encoding="utf8") as html:
        content = html.readlines()
        html.close()
    category = ""
    category_num = 0
    for i in range(0, len(content)):
        if content[i].startswith('<tr><th colspan="15" class="mediumhead"><a href='):
            category = content[i].split('"')[7]
            category = str(category_num) + '_' + category
            category_num = category_num + 1
        if content[i].startswith('<tr><td class="rchars">'):
            parse_emoji(content[i:i + 15], category)
    return


def main():
    fp_manu = settings.EMOJI_IMAGES_PATH + '/by_manufacturer/'
    fp_cat = settings.EMOJI_IMAGES_PATH + '/by_category/'
    fp_manu_fo = settings.EMOJI_IMAGES_PATH + '/by_manufacturer_faces_only/'
    fp_cat_fo = settings.EMOJI_IMAGES_PATH + '/by_category_faces_only/'

    if not os.path.isdir(fp_manu):
        os.mkdir(fp_manu)
    if not os.path.isdir(fp_cat):
        os.mkdir(fp_cat)
    if not os.path.isdir(fp_manu_fo):
        os.mkdir(fp_manu_fo)
    if not os.path.isdir(fp_cat_fo):
        os.mkdir(fp_cat_fo)

    parse_emoji_list()
    return


if __name__ == '__main__':
    main()
