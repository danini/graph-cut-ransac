#ifndef PSO_H_
#define PSO_H_

#include <iostream>
#include <random>
#include <vector>

// CONSTANTS
#define PSO_MAX_SIZE 100 // max swarm size
#define PSO_INERTIA 0.7298 // default value of w (see clerc02)

// === NEIGHBORHOOD SCHEMES ===
#define PSO_NHOOD_GLOBAL 0 // global best topology
#define PSO_NHOOD_RING 1 // ring topology
#define PSO_NHOOD_RANDOM 2 // Random neighborhood topology // **see http://clerc.maurice.free.fr/pso/random_topology.pdf**

// === INERTIA WEIGHT UPDATE FUNCTIONS ===
#define PSO_W_CONST 0
#define PSO_W_LIN_DEC 1

// PSO SOLUTION -- Initialized by the user
typedef struct {

	bool stopping_criterion;
	double error;
	std::vector<double> gbest; // should contain DIM elements!!

} pso_result_t;

// OBJECTIVE FUNCTION TYPE
typedef double (*pso_obj_fun_t)(const std::vector<double>&, int, void *);

// PSO SETTINGS
typedef struct {

	int dim; // problem dimensionality
	std::vector<double> lo; // lower range limit
	std::vector<double> hi; // higher range limit
	double goal; // optimization goal (error threshold)

	int size; // swarm size (number of particles)
	int print_every; // ... N steps (set to 0 for no output)
	int steps; // maximum number of iterations
	int step; // current PSO step
	double c1; // cognitive coefficient
	double c2; // social coefficient
	double w_max; // max inertia weight value
	double w_min; // min inertia weight value

	int clamp_pos; // whether to keep particle position within defined bounds (TRUE)
	// or apply periodic boundary conditions (FALSE)
	int nhood_strategy; // neighborhood strategy (see PSO_NHOOD_*)
	int nhood_size; // neighborhood size 
	int w_strategy; // inertia weight strategy (see PSO_W_*)

	long seed;

	unsigned stopping_criterion_window; // moving average window for stopping criterion
	double stopping_criterion_treshold;

} pso_settings_t;

template<typename T>
void initMatrix(std::vector<std::vector<T>> &m, int h, int w) { m.resize(h); for(int i = 0; i<h; ++i) m[i].resize(w, (T)0); }

typedef void (*Inform_Fun)(std::vector<std::vector<int>>&, std::vector<std::vector<double>>&, std::vector<std::vector<double>>&, std::vector<double>&, std::vector<double>&, int, pso_settings_t);

// return the swarm size based on dimensionality
int pso_calc_swarm_size(int dim);

// set the default PSO settings
void pso_set_default_settings(pso_settings_t &settings);


// minimize the provided obj_fun using PSO with the specified settings
// and store the result in *solution
void pso_solve(pso_obj_fun_t obj_fun, void *obj_fun_params,
			   pso_result_t *solution, pso_settings_t &settings);



#endif // PSO_H_
