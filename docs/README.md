# Interacted Panel VAR Toolbox

## What is this?
	Toolbox that estimates a panel VAR allowing coefficients to vary 
	determinstically with individual characteristics as employed 
	in Towbin and Weber (2011) ”‘Limits of Floating Exchange Rates: 
	the Role of Foreign Currency Debt and Import Structure”’.
	
	This toolbox is a translation of Matlab code to Python.

## This toolbox includes:
- An equation-by-equation OLS VAR estimator for (unbalanced) panels with deterministically varying coefficients. 
- An impulse response function generating file which accounts for the presence of interaction terms.
- A generator of bootstrapped parametric or non-parametric confidence intervals for the IRFs.
- A simple program which graphs impulse responses. 
- A manual.pdf documentation file.

## How to use:
- Prepare data in excel format: All data that is used in any of the files has to be in a standardized order. 
  Each unit (country)needs to have the same time length, if units have different length in the original data missing
  observations (NaN) need to be used to generate identical time length. The data needs to be sorted in the sense that the
  observations 1 to T are of unit one, T+1 to 2*T of unit 2 and so on.
- Define exogenous interacted variables.
- Define a restriction matrix with zeros and ones defining for which equation which interaction and variable 
  a restriction is supposed to be applied (ones leading to an exclusion of the interaction/or variable).
- Use IPVARexample script as a template for developing your script.
- Change directory to src folder and run this script in a command prompt, for example: python IPVARexample.py 
- The produced graphs will be placed in a graphs folder.

## Prerequisites:
- Prior to using this toolbox please install ”‘numpy”’, ”‘scipy”’, ”‘pandas”’ and ”‘matplotlib”’ packages. 

## Who should I contact for questions?
* Alexei Goumilevski agoumilevski@imf.org
* Sebastian Weber SWeber@imf.org
