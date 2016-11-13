#pragma once
#include "NeuralNetwork.h"
using namespace NeuralNetworkLibrary;

// Keeping network initializing parameters
struct NetworkProperty
{
	vector<int> layerSizes;
	vector<TransferFunction> transferFunctions;
	NetworkProperty() {};
	NetworkProperty(vector<int> _layerSizes, vector<TransferFunction> _transferFunctions)
	{
		layerSizes = _layerSizes;
		transferFunctions = _transferFunctions;
	}
};

class Creature
{
private:
	NeuralNetwork* Network;
	double fitness;
	// Run network and return F_measure
	typedef double(*TestNetwork)(NeuralNetwork* _Network);

public:
	Creature()
	{
	}

	void GenerateNewNetwork(NetworkProperty networkProperty)
	{
		Network = new NeuralNetwork(networkProperty.layerSizes, networkProperty.transferFunctions);
	}

	void SetNetwork(NeuralNetwork* _Network)
	{
		Network = _Network;
	}

	NeuralNetwork* GetNetwork()
	{
		return Network;
	}

	double GetFitness()
	{
		return fitness;
	}

	bool operator<(const Creature* b)
	{
		return fitness < b->fitness;
	}

	void Run(TestNetwork testFunction)
	{
		fitness = (*testFunction)(Network);
	}
};