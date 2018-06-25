echo "cleaning PFNET ..."
find . -name \*~ -delete
find ./pfnet -name \*.so -delete
find ./pfnet -name \*.pyc -delete
find ./pfnet -name __pycache__ -delete
find ./tests -name \*.pyc -delete
find ./tests -name __pycache__ -delete
find ./examples -name \*.pyc -delete
find . -name libpfnet* -delete
rm -rf PFNET.egg-info
rm -f ./pfnet/cpfnet.c
rm -f ./pfnet/cpfnet.h
rm -rf build
rm -rf dist
rm -rf lib/pfnet
