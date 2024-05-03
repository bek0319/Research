import unittest

loader = unittest.TestLoader()
tests = loader.discover("utest\.")
print(tests)
testRunner = unittest.runner.TextTestRunner(verbosity=2)
testRunner.run(tests)

