import base64
import copy
import io
import unittest
from unittest.mock import patch

from PIL import Image
from dllm.vision_messages import decode_image_messages


def image_part(color='red',format_name='PNG',size=(8,6)):
    buffer=io.BytesIO()
    Image.new('RGB',size,color).save(buffer,format=format_name)
    mime='png' if format_name=='PNG' else 'jpeg'
    return dict(type='image_url',image_url=dict(url=f'data:image/{mime};base64,'+
        base64.b64encode(buffer.getvalue()).decode('ascii')))


class VisionMessageTests(unittest.TestCase):
    def test_order_identity_and_input_preservation(self):
        messages=[dict(role='user',content=[dict(type='text',text='Compare '),image_part('red'),
                                           dict(type='text',text=' with '),image_part('blue')])]
        before=copy.deepcopy(messages)
        result=decode_image_messages(messages)
        self.assertEqual(messages,before)
        self.assertEqual(result.images[0][0,0].tolist(),[255,0,0])
        self.assertEqual(result.images[1][0,0].tolist(),[0,0,255])
        self.assertNotEqual(*result.identities)
        self.assertEqual(result.messages[0]['content'].count('<|image_pad|>'),2)

    def test_byte_limit_before_image_decode(self):
        with patch('PIL.Image.open') as opened:
            with self.assertRaisesRegex(ValueError,'byte budget'):
                decode_image_messages([dict(role='user',content=[image_part()])],max_bytes=1)
            opened.assert_not_called()

    def test_count_and_combined_pixel_limits(self):
        messages=[dict(role='user',content=[image_part(),image_part()])]
        with self.assertRaisesRegex(ValueError,'count limit'):decode_image_messages(messages,max_images=1)
        with self.assertRaisesRegex(ValueError,'pixel budget'):decode_image_messages(messages,max_pixels=80)

    def test_invalid_urls_and_base64(self):
        for url in ('https://example.com/image.png','file:///tmp/image.png','data:image/png;base64,???'):
            with self.subTest(url=url),self.assertRaises(ValueError):
                decode_image_messages([dict(role='user',content=[dict(type='image_url',image_url=dict(url=url))])])

    def test_format_and_role_validation(self):
        part=image_part()
        part['image_url']['url']=part['image_url']['url'].replace('image/png','image/jpeg')
        with self.assertRaisesRegex(ValueError,'format mismatch'):
            decode_image_messages([dict(role='user',content=[part])])
        with self.assertRaises(ValueError):decode_image_messages([dict(role='system',content=[image_part()])])
        part=image_part();part['image_url']['detail']='low'
        with self.assertRaises(ValueError):decode_image_messages([dict(role='user',content=[part])])

    def test_jpeg_and_text_only(self):
        result=decode_image_messages([dict(role='user',content=[image_part(format_name='JPEG')])])
        self.assertEqual(tuple(result.images[0].shape),(6,8,3))
        result=decode_image_messages([dict(role='user',content='Hello')])
        self.assertEqual(result.images,())
        self.assertEqual(result.messages,[dict(role='user',content='Hello')])


if __name__=='__main__':unittest.main()
