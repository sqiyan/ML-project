# SUTD 50.007 ML-Project
*Seah Qi Yan 1004628*  |  *Ong Kah Yuan Joel 1004366*

## How To Use 

The main file to be run is `HMM.py`. There are several arguments that can be passed to it:

~~~
Arguments:
  -h, --help       show the help message and exit
  --file FILE      Which file to run on. C for chinese, E for english and S
                   for SG. 
                   Usage: --file E
  
  --part PART      Which part to do: 2, 3, 4, 5. 
  				 Usage: --part 2
  
  --action ACTION  train or eval. Train updates the parameters and saves it to  				 a pickle file. Eval will write to dev.{part} in the 					 respective language folders.
  				 Usage: --action train
  				 
Example: 
	To train on part 3 for english, use
		./HMM.py --file E --part 3 --action train
		
	Then, to evaluate, run
		./HMM.py --file E --part 3 --action eval
	
	The output will be written to /EN/dev.p2.out
~~~

Parameters need to be trained to the pickle files before evaluation can be done. The full sequence of commands that need to be run for each part are as follows:

### Part 2

```
PART 2
EN:
./HMM.py --file E --part 2 --action train
./HMM.py --file E --part 2 --action eval

CN:
./HMM.py --file C --part 2 --action train
./HMM.py --file C --part 2 --action eval

SG:
./HMM.py --file S --part 2 --action train
./HMM.py --file S --part 2 --action eval
```

## Part 3

```
PART 3
EN:
./HMM.py --file E --part 3 --action train
./HMM.py --file E --part 3 --action eval

CN:
./HMM.py --file C --part 3 --action train
./HMM.py --file C --part 3 --action eval

SG:
./HMM.py --file S --part 3 --action train
./HMM.py --file S --part 3 --action eval
```

## Part 4

```
./HMM.py --file E --part 4 --action train
./HMM.py --file E --part 4 --action eval
```

## Part 5

```
Lmao ddp things
```

