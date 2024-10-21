In dataset,there are four files,to generating and spliting dataset.

Such as ,if you want to generate MNIST dataset,you can input"python generate_MNIST.py noniid - dir" in cmd."dir" is Practical heterogeneous setting, meanwile "pat" is Pathological heterogeneous setting.

After spliting dataset,you can run FedHG,start by switching directories"cd./System",and inputing"python main.py -data MNIST -nb 10 -m cnn -gr 200 -did 0 -algo FedHG".
