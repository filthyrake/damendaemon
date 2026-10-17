import unittest
from damendaemon.management import command


class ManagementValidationTests(unittest.TestCase):
    def test_supported_commands(self):
        self.assertEqual(command('status')['action'],'status')
        self.assertEqual(command('remove', 'a')['action'], 'remove')
        self.assertEqual(command('register','a','/local/model')['path'],'/local/model')
        self.assertTrue(command('unload','a',release_host=True)['release_host'])

    def test_invalid_commands_fail_before_admission(self):
        for args in (('unknown',),('remove',),('remove', 'a', '/path'),('remove', 'a', None, True),('load',),('load',''),('load','a','/path'),
                     ('status','a'),('register','a'),('register','a','\x00'),
                     ('unload','a',None,1),('load','a',None,True),('register','a','/path',True)):
            with self.subTest(args=args),self.assertRaises(ValueError):command(*args)

    def test_per_model_quantization_validation(self):
        self.assertEqual(command('register','a','/model',quantization='q8_0')['quantization'],'q8_0')
        self.assertEqual(command('register','q4','/model',quantization='q4_0')['quantization'],'q4_0')
        for action,value in (('load','none'),('register','int4'),('register',True)):
            with self.assertRaises(ValueError):
                command(action,'a','/model' if action=='register' else None,quantization=value)
