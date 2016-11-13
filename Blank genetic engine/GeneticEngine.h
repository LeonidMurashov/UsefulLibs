#pragma once
#include <vector>
#include "Creature.h"

using namespace NeuralNetworkLibrary;
using namespace std;

class GeneticEngine
{
	// Run network and return F_measure
	typedef double(*TestNetwork)(NeuralNetwork* _Network);
	TestNetwork testFunction;

	// Keeping network initializing parameters
	NetworkProperty networkProperty;
public:

	GeneticEngine(TestNetwork _testFunction, NetworkProperty prop);

	const int Population_Size = 400;// One generation size
	const int Roulett_Size = 5000;	// Count of segments of Roulett, should be much grater than creatures.size()
									///const int Population_Size = 400;// First generation size

	// Main circle called every iteration
	void GeneticCircle();

	// Create start generation
	void CreateFirstGeneration();

	// Spawns current generation
	void Run();

	// Remove previous generation
	void Remove();

	// Run "roulett" randomizer - return a half of creatures (Selection Function)
	vector<Creature*> RunRoulett();

	// Crossing Function
	void Selection(vector<Creature*> halfFinal);

	// Merge Networks data
	Creature* BreedingFunction(Creature* creature1, Creature* creature2);

	// Merge function, which allow to control merging networks process 
	static double MergeFunction(double a, double b);									// Merge weights and biases

	vector<Creature*> creatures;	// TArray is safe version of std::vector
	vector<Creature*> oldCreatures;

	int iteration;

};
