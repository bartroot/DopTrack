import logging
import sys


#  logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(name)-20s %(levelname)-8s %(message)-s', level=logging.DEBUG)
