import unittest

class TestBasicFunction(unittest.TestCase):
	
	def setUp(self):
		pass
	
	def test_0(self):
		self.assertTrue(1 == 1)
 
if __name__ == '__main__':
	unittest.main()