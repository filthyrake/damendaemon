import unittest
from damendaemon.pages import PagePool, PageCapacityError


class PagePoolTests(unittest.TestCase):
    def test_fragmentation_reuse_and_isolation(self):
        pool=PagePool(128,4,128)
        pool.resize_many([(0,33),(1,17),(2,16)])
        original=list(pool.pages[0])
        pool.release(1)
        pool.resize_many([(3,48)])
        self.assertEqual(pool.pages[0],original)
        owned=[page for pages in pool.pages for page in pages]
        self.assertEqual(len(owned),len(set(owned)))
        self.assertEqual(set(owned)|set(pool.free),set(range(8)))
        self.assertEqual(pool.used_tokens,112)
        self.assertNotEqual(pool.pages[3],sorted(pool.pages[3]))

    def test_failed_multislot_admission_is_atomic(self):
        pool=PagePool(64,2,128)
        pool.resize_many([(0,32)])
        before=([p.copy() for p in pool.pages],pool.free.copy(),pool.revision)
        with self.assertRaises(PageCapacityError):
            pool.resize_many([(0,16),(1,64)],replace=True)
        self.assertEqual(before,(pool.pages,pool.free,pool.revision))
        pool.resize_many([(0,16),(1,48)],replace=True)
        self.assertEqual(pool.used_tokens,64)

    def test_decode_growth_and_idempotent_release(self):
        pool=PagePool(64,2,33)
        pool.resize_many([(0,16)])
        page=pool.pages[0][0]
        pool.resize_many([(0,17)])
        self.assertEqual(pool.pages[0][0],page)
        pool.resize_many([(0,33)])
        self.assertEqual(len(pool.pages[0]),3)
        pool.release(0)
        pool.release(0)
        self.assertEqual(pool.used_tokens,0)
        self.assertEqual(len(pool.free),4)

    def test_invalid_requests_do_not_mutate(self):
        pool=PagePool(64,2,64)
        for requests in ([(0,1),(0,2)],[(2,1)],[(0,65)],[(True,1)],[(0,-1)]):
            with self.subTest(requests=requests),self.assertRaises(ValueError):
                pool.resize_many(requests)
            self.assertEqual(pool.used_tokens,0)
        for tokens in (0,31,16,True):
            with self.assertRaises(ValueError):
                PagePool(tokens,2,64)
