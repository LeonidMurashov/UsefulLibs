#include <iostream>
#include "Creature.h"
#include "GeneticEngine.h"
using namespace std;

int main()
{
	GeneticEngine *engine = new GeneticEngine();
	int iteration = -1;

	while (true)
	{
		iteration++;
		engine->GeneticCircle();
		cout << iteration << endl;
	}
}