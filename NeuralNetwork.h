
/*
*
*	Author: Leonid Murashov 30.05.2016
*
*	Neural network consists of classes(network, layer and neuron),
*	All neurons have connetions with all neurons of previous layer,
*	Several transfer functions,
*	Standart deviation random generator(Gaussian),
*	Merge networks method with passed to merge functions as parameters.
*
*/

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string>
#include <vector>
//using namespace std;

namespace NeuralNetworkLibrary2
{
	//#define Length(a) (sizeof(a) / sizeof(a[0]))	// Returns the length of the array (!!! only for static !!!)
#define randDouble() ((double)(rand() / (double)RAND_MAX))	// Returns a random double from 0 to 1

#define NORMAL_HIDDEN_LAYER_SIZE 30
#define NORMAL_HIDDEN_LAYERS_COUNT 1

#define TRANSFER_FUNCTIONS_COUNT 5


#pragma region Transfer functions and their derivative

	enum TransferFunction
	{
		None,
		Linear,
		Sigmoid,
		Gaussian,
		RationalSigmoid
	};

	class TransferFunctions
	{
	public:
		static double Evaluate(TransferFunction tFunc, double input)
		{
			switch (tFunc)
			{
			case TransferFunction::Sigmoid:
				return sigmoid(input);
			case TransferFunction::Linear:
				return linear(input);
			case TransferFunction::Gaussian:
				return gaussian(input);
			case TransferFunction::RationalSigmoid:
				return rationalsigmoid(input);
			case TransferFunction::None:
			default:
				return 0.0;
			}
		}
		static double EvaluateDerivative(TransferFunction tFunc, double input)
		{
			switch (tFunc)
			{
			case TransferFunction::Sigmoid:
				return sigmoid_derivative(input);
			case TransferFunction::Linear:
				return linear_derivative(input);
			case TransferFunction::Gaussian:
				return gaussian_derivative(input);
			case TransferFunction::RationalSigmoid:
				return rationalsigmoid_derivative(input);
			case TransferFunction::None:
			default:
				return 0.0;
			}
		}

		/* Transfer functions declaration*/
	private:
		static double sigmoid(double x)
		{
			return 1.0 / (1.0 + exp(-x));
		}
		static double sigmoid_derivative(double x)
		{
			return sigmoid(x) * (1 - sigmoid(x));
		}

		static double linear(double x)
		{
			return x;
		}
		static double linear_derivative(double x)
		{
			return 1;
		}

		static double gaussian(double x)
		{
			return exp(-pow(x, 2));
		}
		static double gaussian_derivative(double x)
		{
			return -2.0 * x * gaussian(x);
		}

		static double rationalsigmoid(double x)
		{
			return x / (1.0 + sqrt(1.0 + x * x));
		}
		static double rationalsigmoid_derivative(double x)
		{
			double val = sqrt(1.0 + x * x);
			return 1.0 / (val * (1 + val));
		}
	};

#pragma endregion

	class Gaussian
	{

	public:

		static double GetRandomGaussian()
		{
			return GetRandomGaussian(0.0, 1.0);
		}
		static double GetRandomGaussian(double mean, double stddev)
		{
			double rVal1, rVal2;

			GetRandomGaussian(mean, stddev, rVal1, rVal2);

			return rVal1;
		}
		static void GetRandomGaussian(double mean, double stddev, double &val1, double &val2)
		{
			double u, v, s, t;

			do
			{
				u = 2 * randDouble() - 1;
				v = 2 * randDouble() - 1;
			} while (u * u + v * v > 1 || (u == 0 && v == 0));

			s = u * u + v * v;
			t = sqrt((-2.0 * log(s)) / s);

			val1 = stddev * u * t + mean;
			val2 = stddev * v * t + mean;
		}
	};

	enum LayerPosition { INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER };

	/*Some defines for MergeFunctions*/
	// Merge weights and biases
	typedef double(*Merge1)(double a, double b);
	// Merge layers and neurons count
	typedef int(*Merge2)(int a, int b);
	// Merge TransferFunctions
	typedef TransferFunction(*Merge3)(TransferFunction a, TransferFunction b);

	class Layer
	{
		// Nested class to prevent cyclic inclusion
#pragma region Neuron class

		class Neuron
		{
		private:
			Layer* parentLayer;
			double output;

			// For input layers(is not used by hidden and output layers)
			double inputData;
			// For hidden and output layers(is not used by input layers)
			std::vector<double> weights;
			double bias;

		public:
			// Default constructor
			Neuron(Layer* _parentLayer)
			{
				parentLayer = _parentLayer;
				if (parentLayer->transferFunction != TransferFunction::None) // If neuron is not on input
				{
					for (int i = 0; i < parentLayer->previousLayer->neurons.size(); i++)
						weights.push_back(Gaussian::GetRandomGaussian());
					bias = Gaussian::GetRandomGaussian();
				}
			}
			// Merge constructor
			Neuron(Layer* _parentLayer, std::vector<double> _weights, double _bias)
			{
				parentLayer = _parentLayer;
				weights = _weights;
				bias = _bias;
			}

			void Run()
			{
				if (parentLayer->transferFunction != TransferFunction::None) // If neuron is not on input
				{
					double sum = 0;
					for (int i = 0; i < parentLayer->previousLayer->neurons.size(); i++)
						sum += weights[i] * parentLayer->previousLayer->neurons[i]->GetOutput();

					output = TransferFunctions::Evaluate(parentLayer->transferFunction, sum + bias);
				}
				else
					output = inputData;
			}

			double GetOutput() { return output; }

			void SetInputData(double _inputData) { inputData = _inputData; }

		private:
			static void CastWeights(std::vector<double> &a, Layer *_parentLayer)
			{
				while (a.size() < _parentLayer->previousLayer->neurons.size())
					a.push_back(Gaussian::GetRandomGaussian());

				if (a.size() > _parentLayer->previousLayer->neurons.size())
					a.resize(_parentLayer->previousLayer->neurons.size());
			}

		public:
			// MergeFunction() is situated in [GeneticEngine] source file 
			// Cannot be used to create input neuron
			static Neuron* MergeNeurons(Neuron* neuron1, Neuron* neuron2, Layer* _parentLayer, Merge1 MergeFunction)
			{
				if (_parentLayer->transferFunction == TransferFunction::None)
					throw("MergeNeurons function cannot be used to create input neuron");

				std::vector<double> weights1, weights2, finalWeights;
				double bias1, bias2, finalBias;

				// Getting data from neurons 1 and 2
				neuron1->GetInvolvedData(weights1, bias1);
				neuron2->GetInvolvedData(weights2, bias2);

				//Casting weights to be the same demention
				CastWeights(weights1, _parentLayer);
				CastWeights(weights2, _parentLayer);

				// Merging data
				for (int i = 0; i < _parentLayer->previousLayer->neurons.size(); i++)
					finalWeights.push_back((*MergeFunction)(weights1[i], weights2[i]));
				finalBias = (*MergeFunction)(bias1, bias2);

				return new Neuron(_parentLayer, finalWeights, finalBias);
			}

			void GetInvolvedData(std::vector<double> &_weights, double &_bias)
			{
				_weights = weights;
				_bias = bias;
			}

			~Neuron()
			{
				weights.clear();
			}
		};

#pragma endregion

		// Layer class declaration
	public:
		std::vector<Neuron*> neurons;
		TransferFunction transferFunction;
		Layer* previousLayer;

		// Default constructor
		Layer(int size, TransferFunction _transferFunction, Layer *_previousLayer)
		{
			transferFunction = _transferFunction;
			previousLayer = _previousLayer;

			// Create neurons
			for (int i = 0; i < size; i++)
				neurons.push_back(new Neuron(this));
		}

		// Merge constructor
		Layer(std::vector<Neuron*> _neurons, TransferFunction _transferFunction, Layer *_previousLayer)
		{
			neurons = _neurons;
			transferFunction = _transferFunction;
			previousLayer = _previousLayer;
		}

		void Run()
		{
			for (int i = 0; i < neurons.size(); i++)
				neurons[i]->Run();
		}

		// MergeFunction() is situated in [GeneticEngine] source file 
		// Layer* previosActualLayer is needed for creating layer, but if
		//		layer is on input can be NULL
		static Layer* MergeLayers(Layer* lay1, Layer* lay2, Layer* previosActualLayer, LayerPosition layerPosition,
			Merge1 MergeFunction, Merge2 MergeFunctionInt,
			Merge3 MergeFunctionTransfer)
		{
			std::vector<Neuron*> _neurons;
			TransferFunction _transferFunction = (*MergeFunctionTransfer)(lay1->transferFunction, lay2->transferFunction);
			int size;

			// Set demention of layer			
			if (layerPosition == LayerPosition::HIDDEN_LAYER)
				size = (*MergeFunctionInt)(lay1->neurons.size(), lay2->neurons.size());
			else
				size = lay1->neurons.size();

			// Neurons will be merged and set later
			Layer *finalLayer = new Layer(0, _transferFunction, previosActualLayer);

			// Actions for not input layer 
			if (previosActualLayer != NULL)
			{
				CastLayer(lay1, size);
				CastLayer(lay2, size);
				for (int i = 0; i < size; i++)
					_neurons.push_back(Neuron::MergeNeurons(lay1->neurons[i], lay2->neurons[i], finalLayer, MergeFunction));
			}
			else // Actions in case of input layer
			{
				for (int i = 0; i < size; i++)
					_neurons.push_back(new Neuron(finalLayer));
			}

			finalLayer->neurons = _neurons;
			return finalLayer;
		}

		static Layer* GetRandomLayer(Layer* _previousLayer)
		{
			int size;
			do
			{
				size = round(Gaussian::GetRandomGaussian(NORMAL_HIDDEN_LAYER_SIZE, (double)NORMAL_HIDDEN_LAYER_SIZE / (double)2));
			} while (size < 1);

			return new Layer(size, TransferFunction(rand() % (TRANSFER_FUNCTIONS_COUNT - 2) + 2), _previousLayer);
		}

		~Layer()
		{
			neurons.clear();
		}


	private:
		static void CastLayer(Layer * lay, int size)
		{
			while (lay->neurons.size() < size)
				lay->neurons.push_back(new Neuron(lay));
			if (lay->neurons.size() > size)
				lay->neurons.resize(size);
		}
	};

	class NeuralNetwork
	{
		std::vector<Layer*> layers;

	public:
		// Default constructor
		NeuralNetwork(std::vector<int> layerSizes, std::vector <TransferFunction> transferFunctions)
		{
			// Validate the input data
			if (transferFunctions.size() != layerSizes.size() || transferFunctions[0] != TransferFunction::None)
				throw ("Cannot construct a network with these parameters");

			// Create layers
			for (int i = 0; i < layerSizes.size(); i++)
				layers.push_back(new Layer(layerSizes[i], transferFunctions[i], i == 0 ? NULL : layers[i - 1]));
		}

		// Merge constructor
		NeuralNetwork(std::vector<Layer*> _layers)
		{
			// Validate the input data
			if (_layers.size()<2)
				throw ("Cannot construct a network with these parameters");

			// Set layers
			layers = _layers;
		}

		std::vector<double> Run(std::vector<double> input)
		{
			// Make sure we have enough data
			if (input.size() != layers[0]->neurons.size())
				throw ("Input data is not of the correct dimention.");

			for (int i = 0; i < layers[0]->neurons.size(); i++)
				layers[0]->neurons[i]->SetInputData(input[i]);

			// Calculating the result 
			for (int i = 0; i < layers.size(); i++)
				layers[i]->Run();

			// Pushing the result to std::vector
			std::vector<double> output;
			for (int i = 0; i < layers[layers.size() - 1]->neurons.size(); i++)
				output.push_back(layers[layers.size() - 1]->neurons[i]->GetOutput());

			return output;
		}

		std::vector<Layer*> GetLayers() { return layers; }

	private:
		static void CastNetworkLayers(std::vector<Layer*> &a, int size)
		{
			if (size < 2)
				throw("Casting network layers failed");

			Layer* outputLayer = a[a.size() - 1];
			a.pop_back();

			while (a.size() < size - 1)
				a.push_back(Layer::GetRandomLayer(a[a.size() - 1]));
			if (a.size() > size - 1)
				a.resize(size - 1);

			a.push_back(outputLayer);
		}

	public:
		// MergeFunction() is situated in [GeneticEngine] source file
		static NeuralNetwork* MergeNetworks(NeuralNetwork* net1, NeuralNetwork* net2,
			Merge1 MergeFunction, Merge2 MergeFunctionInt,
			Merge3 MergeFunctionTransfer)
		{
			if (!MergeFunction || !MergeFunctionInt || !MergeFunctionTransfer)
				throw ("NULL MergeFuction recieved");

			std::vector<Layer*> layers1 = net1->GetLayers(),
				layers2 = net2->GetLayers(),
				finalLayers;
			int size = (*MergeFunctionInt)(layers1.size(), layers2.size());

			// Casting layers to be the same demention
			CastNetworkLayers(layers1, size);
			CastNetworkLayers(layers2, size);

			for (int i = 0; i < size; i++)
			{
				LayerPosition layerPosition;
				if (i == 0) layerPosition = LayerPosition::INPUT_LAYER;
				else if (i == size - 1) layerPosition = LayerPosition::OUTPUT_LAYER;
				else layerPosition = LayerPosition::HIDDEN_LAYER;

				finalLayers.push_back(Layer::MergeLayers(layers1[i], layers2[i], i == 0 ? NULL : finalLayers[i - 1], layerPosition,
					MergeFunction, MergeFunctionInt, MergeFunctionTransfer));
			}

			return new NeuralNetwork(finalLayers);
		}

		static NeuralNetwork* GetRandomNetwork(int inputsCount, int outputsCount)
		{
			std::vector<Layer*> _layers;
			_layers.push_back(new Layer(inputsCount, TransferFunction::None, NULL));
			int size;
			do
			{
				size = round(Gaussian::GetRandomGaussian(NORMAL_HIDDEN_LAYERS_COUNT, (double)NORMAL_HIDDEN_LAYERS_COUNT / (double)2));
			} while (size < 0);

			for (int i = 0; i < size; i++)
				_layers.push_back(Layer::GetRandomLayer(_layers[_layers.size() - 1]));

			_layers.push_back(new Layer(outputsCount, TransferFunction::Linear, _layers[_layers.size() - 1]));
			return new NeuralNetwork(_layers);
		}

		~NeuralNetwork()
		{
			layers.clear();
		}
	};
}