This is the relevant codes of the paper "Fast Semi-supervised Learning on Large Graphs: An Improved Green-function Method".

To run the codes, place the ./data and ./py folders in a new project directory, and then select the experiment function, the dataset, and the method from. /py/main.py. GF(G) and GF(A) are the proposed methods in this paper.

The datasets are stored in the ./data folder. For now, only ./data/Toy and ./data/USPS datasets are provided. If one needs to add additional datasets, a translation code should be implemented and modeled after the output format of ./data/USPS/usps.py and imported in ./data/data_input.

If one only needs an implement, ./py/method.py, ./py/question.py, and ./py/BKHK.py are necesarry. The data interface should be modified in the question class, and then the get_indicator function of the method class will give out the soft labels and runtime.
