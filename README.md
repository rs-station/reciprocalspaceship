# reciprocalspaceship
Tools for exploring reciprocal space

`reciprocalspaceship` provides a Pandas-style dataframe object for
analyzing and manipulating reflection data from crystallography
experiments.

## Getting Started 

We have some documentation, which can be viewed by cloning the repository.
Someday, we will have this hosted, but not today...

```
git clone https://github.com/Hekstra-Lab/reciprocalspaceship.git
cd reciprocalspaceship

# This is sloppy... but just open docs/_build/html/index.html in browser
if command -v xdg-open > /dev/null 2>&1
then
    xdg-open docs/_build/html/index.html
elif command -v open > /dev/null 2>&1
then
    open docs/_build/html/index.html
else
    echo "I don't know what browser you use"
fi

```
