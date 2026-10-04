import unittest
from dllm.management import command


class ManagementValidationTests(unittest.TestCase):
    def test_supported_commands(self):
        self.assertEqual(command('status')['action'],'status')
        self.assertEqual(command('register','a','/local/model')['path'],'/local/model')
        self.assertTrue(command('unload','a',release_host=True)['release_host'])

    def test_invalid_commands_fail_before_admission(self):
        for args in (('unknown',),('load',),('load',''),('load','a','/path'),
                     ('status','a'),('register','a'),('register','a','\x00'),
                     ('unload','a',None,1),('load','a',None,True),('register','a','/path',True)):
            with self.subTest(args=args),self.assertRaises(ValueError):command(*args)
