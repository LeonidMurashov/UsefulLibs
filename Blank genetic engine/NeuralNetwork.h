
/*
*
*	Author: Leonid Murashov 29.05.2016
*
*	Version: 1.2	
*	Neural network consists of tons of arrays,
*	All neurons have connetions with all neurons of previous layer,
*	Several transfer functions,
*	Error back-propagation method of training,
*	Normal distribution random generator(Gaussian),
*	Save and load from file methods,
*	Merge biases and weights function(are used in genetic algorithm).
*/
#pragma once
#include <stdlib.h>     /* rand */
#include <time.h>       /* time */
#include <math.h>
#include <string>
#include <vector>
#include <assert.h>
#include "tinyxml2.h"
using namespace std;

namespace NeuralNetworkLibrary
{
	using namespace tinyxml2;

//#define Length(a) (sizeof(a) / sizeof(a[0]))	// Returns the length of the array (!!! only for static !!!)
#define randDouble() ((double)(rand() / (double)RAND_MAX))	// Returns a random double from 0 to 1

#pragma region Transfer functions and their derivative

	enum TransferFunction
	{
		None,
		Sigmoid,
		Linear,
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

	/* Define for MergeFunction*/
	// Merge weights and biases
	typedef double(*Merge)(double a, double b);

	class NeuralNetwork
	{

#pragma region Constructors and destructors

	public:
		NeuralNetwork(vector<int> layerSizes, vector <TransferFunction> transferFunctions)
		{
			//Validate the input data
			if (transferFunctions.size() != layerSizes.size() || transferFunctions[0] != TransferFunction::None)
			{
				assert(false && "Cannot construct a network with these parameters");
			}

			//Initialize network layers
			layerCount = (uint32_t)layerSizes.size() - 1;
			inputSize = layerSizes[0];

			layerSize.resize(layerCount);
			for (int i = 0; i < layerCount; i++)
				layerSize[i] = layerSizes[i + 1];

			transferFunction.resize(layerCount);
			for (int i = 0; i < layerCount; i++)
				transferFunction[i] = transferFunctions[i + 1];

			//Start dementioning arrays
			bias.resize(layerCount);
			previousBiasDelta.resize(layerCount);
			delta.resize(layerCount);
			layerOutput.resize(layerCount);
			layerInput.resize(layerCount);

			weight.resize(layerCount);
			previousWeightDelta.resize(layerCount);

			//Fill 2 dimentional arrays
			for (int l = 0; l < layerCount; l++)
			{
				bias[l].resize(layerSize[l]);
				previousBiasDelta[l].resize(layerSize[l]);
				delta[l].resize(layerSize[l]);
				layerOutput[l].resize(layerSize[l]);
				layerInput[l].resize(layerSize[l]);

				weight[l].resize(l == 0 ? inputSize : layerSize[l - 1]);
				previousWeightDelta[l].resize(l == 0 ? inputSize : layerSize[l - 1]);

				for (int i = 0; i<(l == 0 ? inputSize : layerSize[l - 1]); i++)
				{
					weight[l][i].resize(layerSize[l]);
					previousWeightDelta[l][i].resize(layerSize[l]);
				}
			}

			// Initialize the weights
			for (int l = 0; l<layerCount; l++)
			{
				for (int j = 0; j<layerSize[l]; j++)
				{
					bias[l][j] = Gaussian::GetRandomGaussian();
					previousBiasDelta[l][j] = 0.0;
					layerOutput[l][j] = 0.0;
					layerInput[l][j] = 0.0;
					delta[l][j] = 0.0;
				}

				for (int i = 0; i< (l == 0 ? inputSize : layerSize[l - 1]); i++)
				{
					for (int j = 0; j<layerSize[l]; j++)
					{
						weight[l][i][j] = Gaussian::GetRandomGaussian();
						previousWeightDelta[l][i][j] = 0.0;
					}
				}
			}
		}

		~NeuralNetwork()
		{
			Name.clear();

			layerSize.clear();
			transferFunction.clear();

			layerOutput.clear();
			layerInput.clear();
			bias.clear();
			delta.clear();
			previousBiasDelta.clear();

			weight.clear();
			previousWeightDelta.clear();
		}

#pragma endregion

#pragma region Methods

	public:
		void Run(vector <double> input, vector <double> &output)
		{
			//Make sure we have enough data
			if (input.size() != inputSize)
				assert (false && "Input data is not of the correct dimention.");

			//Dimention 
			output = vector<double>();
			output.resize(layerSize[layerCount - 1]);
			
			/* Run the network */
			for (int l = 0; l < layerCount; l++)
			{
				for (int j = 0; j < layerSize[l]; j++)
				{
					double sum = 0.0;
					for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
						sum += weight[l][i][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);

					sum += bias[l][j];
					layerInput[l][j] = sum;

					layerOutput[l][j] = TransferFunctions::Evaluate(transferFunction[l], sum);
				}
			}

			//Copy the output to the output array
			for (int i = 0; i < layerSize[layerCount - 1]; i++)
				output[i] = layerOutput[layerCount - 1][i];
		}

		double Train(vector<double> input, vector<double> &desired, double TrainingRate, double Momentum)
		{
			// Parameter Validation
			if (input.size() != inputSize)
				assert (false && "Ivalid input parameter", "input");
			if (desired.size() != layerSize[layerCount - 1])
				assert(false && "Invalid input parameter", "desired");

			// Local variable
			double error = 0.0, sum = 0.0, weightDelta = 0.0, biasDelta = 0.0;

			vector <double> output;
			output.resize(layerSize[layerCount - 1]);

			//Run the network
			Run(input, output);

			// Back-propagate the error
			for (int l = layerCount - 1; l >= 0; l--)
			{
				//Output layer
				if (l == layerCount - 1)
				{
					for (int k = 0; k < layerSize[l]; k++)
					{
						delta[l][k] = output[k] - desired[k];
						error += pow(delta[l][k], 2);
						delta[l][k] *= TransferFunctions::EvaluateDerivative(transferFunction[l],
							layerInput[l][k]);
					}
				}
				else //Hidden layer
				{
					for (int i = 0; i < layerSize[l]; i++)
					{
						sum = 0.0;
						for (int j = 0; j < layerSize[l + 1]; j++)
						{
							sum += weight[l + 1][i][j] * delta[l + 1][j];
						}
						sum *= TransferFunctions::EvaluateDerivative(transferFunction[l], layerInput[l][i]);

						delta[l][i] = sum;
					}
				}
			}

			//Update the weights and biases
			for (int l = 0; l < layerCount; l++)
				for (int i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++)
					for (int j = 0; j<layerSize[l]; j++)
					{
						weightDelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOutput[l - 1][i])
							+ Momentum * previousWeightDelta[l][i][j];
						weight[l][i][j] -= weightDelta;

						previousWeightDelta[l][i][j] = weightDelta;
					}
			for (int l = 0; l < layerCount; l++)
				for (int i = 0; i<layerSize[l]; i++)
				{
					biasDelta = TrainingRate * delta[l][i];
					bias[l][i] -= biasDelta + Momentum * previousBiasDelta[l][i];

					previousBiasDelta[l][i] = biasDelta;
				}
			return error;
		}

		// Merge two neural networks by merging weights and biases with MergeFunction
		static NeuralNetwork* MergeNetworks(NeuralNetwork* net1, NeuralNetwork* net2, Merge MergeFunction)
		{
			vector<vector<vector<double>>> weight1 = net1->weight,
											weight2 = net2->weight,
											finalWeight;
			vector<vector<double>> bias1 = net1->bias,
									bias2 = net2->bias,
									finalBias;
			try
			{
				finalWeight.resize(weight1.size());
				finalBias.resize(weight1.size());

				for (int i = 0; i < weight1.size(); i++)
				{
					finalWeight[i].resize(weight1[i].size());
					finalBias[i].resize(weight1[i].size());
					for (int j = 0; j < weight1[i].size(); j++)
					{
						finalWeight[i][j].resize(weight1[i][j].size());
						finalBias[i][j] = (*MergeFunction)(bias1[i][j], bias2[i][j]);
						for (int k = 0; k < weight1[i][j].size(); k++)
							finalWeight[i][j][k] = (*MergeFunction)(weight1[i][j][k], weight2[i][j][k]);
					}
				}
				
				vector<int> layerSizes = { net1->inputSize };
				vector<TransferFunction> transferFunctions = { TransferFunction::None };
				layerSizes.insert(layerSizes.end(), net1->layerSize.begin(), net1->layerSize.end());
				transferFunctions.insert(transferFunctions.end(), net1->transferFunction.begin(), net1->transferFunction.end());
				
				NeuralNetwork* finalNetwork = new NeuralNetwork(layerSizes, transferFunctions);
				finalNetwork->weight = finalWeight;
				finalNetwork->bias = finalBias;
				return finalNetwork;
			}
			catch (...)
			{
				return nullptr;
			}
		}

		#define XMLCheckResult(a_eResult) if (a_eResult != tinyxml2::XML_SUCCESS) { return nullptr; }
		#define XMLCheckResultForNull(a_eResult) if(!a_eResult) { return nullptr; }

		/*
		Return a Neural Network created according to file
		*/
		static NeuralNetwork *LoadFromFile(const char* filePath)
		{
			const char* name;
			int layerCount, inputSize;
			vector<int> layerSize;
			vector<TransferFunction> transferFunction;
			vector<vector<double>> bias;
			vector<vector<vector<double>>> weight;

			tinyxml2::XMLDocument xmlDoc;
			XMLCheckResult(xmlDoc.LoadFile(filePath));

			tinyxml2::XMLNode * pRoot = xmlDoc.FirstChild();
			XMLCheckResultForNull(pRoot);
			
			XMLElement * attributesElement = pRoot->FirstChildElement("BasicInfo");
			XMLCheckResultForNull(attributesElement);
			name = attributesElement->Attribute("Name");
			XMLCheckResultForNull(name);
			XMLCheckResult(attributesElement->QueryIntAttribute("layerCount", &layerCount));
			XMLCheckResult(attributesElement->QueryIntAttribute("inputSize", &inputSize));

			// Reading layerSize
			XMLElement *sizesElement = attributesElement->FirstChildElement("LayerSizes");
			XMLCheckResultForNull(sizesElement);
			XMLElement *sizeElement = sizesElement->FirstChildElement("LayerSize");
			XMLCheckResultForNull(sizeElement);
			while (sizeElement)
			{
				int size;
				XMLCheckResult(sizeElement->QueryIntText(&size));
				layerSize.push_back(size);

				sizeElement = sizeElement->NextSiblingElement("LayerSize");
			}

			// Reading tFuncs
			XMLElement *tFuncsElement = sizesElement->NextSiblingElement("TransferFunctions");
			XMLCheckResultForNull(tFuncsElement);
			XMLElement *tFuncElement = tFuncsElement->FirstChildElement("TransferFunction");
			XMLCheckResultForNull(tFuncElement);
			while (tFuncElement)
			{
				int tFunc;
				XMLCheckResult(tFuncElement->QueryIntText(&tFunc));
				transferFunction.push_back((TransferFunction)tFunc);

				tFuncElement = tFuncElement->NextSiblingElement("TransferFunction");
			}

			// Reading biases
			XMLElement *biasesElement = attributesElement->NextSiblingElement("Biases");
			XMLCheckResultForNull(biasesElement);
			XMLElement *preBiasesElement = biasesElement->FirstChildElement("PreBiases");
			XMLCheckResultForNull(preBiasesElement);
			while (preBiasesElement)
			{
				vector<double> preBias;
				XMLElement *biasElement = preBiasesElement->FirstChildElement("Bias");
				XMLCheckResultForNull(biasElement);
				while (biasElement)
				{
					double _bias;
					XMLCheckResult(biasElement->QueryDoubleText(&_bias));
					preBias.push_back(_bias);

					biasElement = biasElement->NextSiblingElement("Bias");
				}
				bias.push_back(preBias);
				preBiasesElement = preBiasesElement->NextSiblingElement("PreBiases");
			}

			//Reading weights
			XMLElement *weightsElement = biasesElement->NextSiblingElement("Weights");
			XMLCheckResultForNull(weightsElement);
			XMLElement *prePreWeightsElement = weightsElement->FirstChildElement("PrePreWeights");
			XMLCheckResultForNull(prePreWeightsElement);
			while (prePreWeightsElement)
			{
				vector<vector<double>> prePreWeights;
				XMLElement *preWeightsElement = prePreWeightsElement->FirstChildElement("PreWeights");
				XMLCheckResultForNull(preWeightsElement);
				while (preWeightsElement)
				{
					vector<double> preWeights;
					XMLElement *weightElement = preWeightsElement->FirstChildElement("Weight");
					XMLCheckResultForNull(weightElement);
					while (weightElement)
					{
						double _weight;
						XMLCheckResult(weightElement->QueryDoubleText(&_weight));
						
						preWeights.push_back(_weight);
						weightElement = weightElement->NextSiblingElement("Weight");
					}

					prePreWeights.push_back(preWeights);
					preWeightsElement = preWeightsElement->NextSiblingElement("PreWeights");
				}

				weight.push_back(prePreWeights);
				prePreWeightsElement = prePreWeightsElement->NextSiblingElement("PrePreWeights");
			}

			// Initializing blank network
			vector<int> _layerSize{ inputSize };
			_layerSize.insert(_layerSize.end(), layerSize.begin(), layerSize.end());
			vector<TransferFunction> _transferFunction{ TransferFunction::None };
			_transferFunction.insert(_transferFunction.end(), transferFunction.begin(), transferFunction.end());
			try {
				// Some checks
				if (weight.size() != layerCount || layerCount != layerSize.size() || bias.size() != layerCount)
					assert(false || "Initializing parameters are not correct!");
				NeuralNetwork *Network = new NeuralNetwork(_layerSize, _transferFunction);
				Network->Name = name;
				Network->bias = bias;
				Network->weight = weight;
				return Network;
			}
			catch (...)
			{
				return nullptr;
			}
		}

		#undef XMLCheckResult
		#define XMLCheckResult(a_eResult) if (a_eResult != tinyxml2::XML_SUCCESS) { return false; }
		
		/*
		Save Network to XML file
		*/
		bool SaveToFile(const char* filePath)
		{
			tinyxml2::XMLDocument xmlDoc;
			XMLCheckResult(xmlDoc.SaveFile(filePath));

			tinyxml2::XMLNode * pRoot = xmlDoc.NewElement("BPNeuralNetwork");
			xmlDoc.InsertFirstChild(pRoot);

			XMLElement *attributesElement = xmlDoc.NewElement("BasicInfo");
			attributesElement->SetAttribute("Name", Name.c_str());
			attributesElement->SetAttribute("layerCount", layerCount);
			attributesElement->SetAttribute("inputSize", inputSize);
			pRoot->InsertFirstChild(attributesElement);

			XMLElement *sizesElement = xmlDoc.NewElement("LayerSizes");
			for (int i = 0; i < layerSize.size(); i++)
			{
				XMLElement *sizeElement = xmlDoc.NewElement("LayerSize");
				sizeElement->SetText(layerSize[i]);
				sizesElement->InsertEndChild(sizeElement);
			}
			attributesElement->InsertEndChild(sizesElement);

			XMLElement *tFuncsElement = xmlDoc.NewElement("TransferFunctions");
			for (int i = 0; i < transferFunction.size(); i++)
			{
				XMLElement *tFuncElement = xmlDoc.NewElement("TransferFunction");
				tFuncElement->SetText(transferFunction[i]);
				tFuncsElement->InsertEndChild(tFuncElement);
			}
			attributesElement->InsertEndChild(tFuncsElement);
			
			// Insert biases
			XMLElement *biasesElement = xmlDoc.NewElement("Biases");
			for (int i = 0; i < bias.size(); i++)
			{
				XMLElement *preBiasesElement = xmlDoc.NewElement("PreBiases");
				for (int j = 0; j < bias[i].size(); j++)
				{
					XMLElement *biasElement = xmlDoc.NewElement("Bias");
					biasElement->SetText(bias[i][j]);
					preBiasesElement->InsertEndChild(biasElement);
				}
				biasesElement->InsertEndChild(preBiasesElement);
			}
			pRoot->InsertEndChild(biasesElement);

			// Insert weights
			XMLElement *weightsElement = xmlDoc.NewElement("Weights");
			for (int i = 0; i < weight.size(); i++)
			{
				XMLElement *prePreWeightsElement = xmlDoc.NewElement("PrePreWeights");
				for (int j = 0; j < weight[i].size(); j++)
				{
					XMLElement *preWeightsElement = xmlDoc.NewElement("PreWeights");
					for (int k = 0; k < weight[i][j].size(); k++)
					{
						XMLElement *weightElement = xmlDoc.NewElement("Weight");
						weightElement->SetText(weight[i][j][k]);
						preWeightsElement->InsertEndChild(weightElement);
					}
					prePreWeightsElement->InsertEndChild(preWeightsElement);
				}
				weightsElement->InsertEndChild(prePreWeightsElement);
			}
			pRoot->InsertEndChild(weightsElement);

			xmlDoc.InsertEndChild(pRoot);
			XMLCheckResult(xmlDoc.SaveFile(filePath));
			return true;
		}
#pragma endregion

#pragma region Public data

	public:
		string Name = "Default";

#pragma endregion

#pragma region Private data

	private:
		int layerCount;
		int inputSize;
		vector<int> layerSize;
		vector<TransferFunction> transferFunction;

		vector<vector<double>> layerOutput;
		vector<vector<double>> layerInput;
		vector<vector<double>> bias;
		vector<vector<double>> delta;
		vector<vector<double>> previousBiasDelta;

		vector<vector<vector<double>>> weight;
		vector<vector<vector<double>>> previousWeightDelta;

#pragma endregion

	};

}