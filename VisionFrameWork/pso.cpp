#include "stdafx.h"
#include <time.h> // for time()
#include <math.h> // for cos(), pow(), sqrt() etc.
#include <float.h> // for DBL_MAX
#include <string.h> // for mem*

#include "pso.h"
#include <queue>
#include <algorithm>

//==============================================================
// calulate swarm size based on dimensionality
int pso_calc_swarm_size(int dim) {
	double size = 10. + 2. * sqrt(dim);
	return (size > PSO_MAX_SIZE ? PSO_MAX_SIZE : (int)size);
}

#pragma region INERTIA WEIGHT UPDATE STRATEGIES
// calculate linearly decreasing inertia weight
double calc_inertia_lin_dec(int step, pso_settings_t settings) {

	int dec_stage = 3 * settings.steps / 4;
	if (step <= dec_stage)
		return settings.w_min + (settings.w_max - settings.w_min) *	\
		(dec_stage - step) / dec_stage;
	else
		return settings.w_min;
}  
#pragma endregion

#pragma region NEIGHBORHOOD (COMM) MATRIX STRATEGIES

// global neighborhood
void inform_global(std::vector<std::vector<int>> &comm, std::vector<std::vector<double>> &pos_nb,
	std::vector<std::vector<double>> &pos_b, std::vector<double> &fit_b,
	std::vector<double> &gbest, int improved,
	pso_settings_t settings)
{
	//std::cout << gbest[0] << " " << gbest[1] << std::endl;
	int i;
	// all particles have the same attractor (gbest)
	// copy the contents of gbest to pos_nb
	for (i=0; i<settings.size; i++)
		std::copy(gbest.begin(), gbest.end(), pos_nb[i].begin());

}

// ===============================================================
// general inform function :: according to the connectivity
// matrix COMM, it copies the best position (from pos_b) of the
// informers of each particle to the pos_nb matrix
void inform(std::vector<std::vector<int>> &comm, std::vector<std::vector<double>> &pos_nb,
	std::vector<std::vector<double>> &pos_b, std::vector<double> &fit_b,
	int improved, pso_settings_t settings)
{
	int i, j;
	int b_n; // best neighbor in terms of fitness
	// for each particle
	for (j=0; j<settings.size; j++) {
		b_n = j; // self is best
		// who is the best informer??
		for (i=0; i<settings.size; i++) {
			// the i^th particle informs the j^th particle
			if (comm[i][j])
				if(fit_b[i] > fit_b[b_n])
					// found a better informer for j^th particle
						b_n = i;
		}
		// copy pos_b of b_n^th particle to pos_nb[j]
		std::copy(pos_b[b_n].begin(), pos_b[b_n].end(), pos_nb[j].begin());
	}
}

// =============
// ring topology
// =============

// topology initialization :: this is a static (i.e. fixed) topology
void init_comm_ring(std::vector<std::vector<int>> &comm, pso_settings_t settings) {
	int i;
	// reset array
	initMatrix(comm, settings.size, settings.size);

	comm[0][0] = 1;

	comm[0][1] = 1;
	comm[0][settings.size-1] = 1;

	comm[settings.size-1][0] = 1;
	comm[settings.size-1][settings.size-2] = 1;

	// choose informers
	for (i=1; i<settings.size-1; i++) {
		// set diagonal to 1
		comm[i][i] = 1;
		// look right
		comm[i][i+1] = 1;
		// look left
		comm[i][i-1] = 1;
	}

}

void inform_ring(std::vector<std::vector<int>> &comm, std::vector<std::vector<double>> &pos_nb,
	std::vector<std::vector<double>> &pos_b, std::vector<double> &fit_b,
	std::vector<double> &gbest, int improved,
	pso_settings_t settings)
{

	// update pos_nb matrix
	inform(comm, pos_nb, pos_b, fit_b, improved, settings);

}

// ============================
// random neighborhood topology
// ============================
void init_comm_random(std::vector<std::vector<int>> &comm, pso_settings_t settings) {

	int i, j, k;
	// reset array
	initMatrix(comm, settings.size, settings.size);

	// choose informers
	for (i=0; i<settings.size; i++) {
		// each particle informs itself
		comm[i][i] = 1;
		// choose kappa (on average) informers for each particle

		static std::minstd_rand g(settings.seed);
		std::uniform_int_distribution<> d(0, settings.size-1);

		for (k=0; k<settings.nhood_size; k++) {
			// generate a random index
			j = d(g);
			// particle i informs particle j
			comm[i][j] = 1;
		}
	}
}

void inform_random(std::vector<std::vector<int>> &comm, std::vector<std::vector<double>> &pos_nb,
	std::vector<std::vector<double>> &pos_b, std::vector<double> &fit_b,
	std::vector<double> &gbest, int improved,
	pso_settings_t settings)
{


	// regenerate connectivity??
	if (!improved)
		init_comm_random(comm, settings);
	inform(comm, pos_nb, pos_b, fit_b, improved, settings);

}

#pragma endregion

//==============================================================
// return default pso settings
void pso_set_default_settings(pso_settings_t &settings) {

	// set some default values
	settings.dim = 2;
	settings.lo = std::vector<double>(settings.dim);
	settings.hi = std::vector<double>(settings.dim);
	std::fill(settings.lo.begin(), settings.lo.end(), -20);
	std::fill(settings.hi.begin(), settings.hi.end(), +20);
	settings.goal = 1e-5;

	settings.size = pso_calc_swarm_size(settings.dim);
	settings.print_every = 1000;
	settings.steps = 100000;
	settings.c1 = 1.496;
	settings.c2 = 1.496;
	settings.w_max = PSO_INERTIA;
	settings.w_min = 0.3;

	settings.clamp_pos = 1;
	settings.nhood_strategy = PSO_NHOOD_RING;
	settings.nhood_size = 5;
	settings.w_strategy = PSO_W_LIN_DEC;

	settings.stopping_criterion_treshold = 1e-3;
	settings.stopping_criterion_window = 10;

	settings.seed = long(time(0));
}

//==============================================================
//                     PSO ALGORITHM
//==============================================================
void pso_solve(pso_obj_fun_t obj_fun, void *obj_fun_params,
			   pso_result_t *solution, pso_settings_t &settings)
{

	// Particles
	std::vector<std::vector<double>> pos, vel, pos_b, pos_nb;
	std::vector<std::vector<int>> comm;

	initMatrix(pos, settings.size, settings.dim); // position matrix
	initMatrix(vel, settings.size, settings.dim);// velocity matrix
	initMatrix(pos_b, settings.size, settings.dim); // best position matrix
	std::vector<double> fit(settings.size); // particle fitness vector
	std::vector<double> fit_b(settings.size); // best fitness vector
	// Swarm
	initMatrix(pos_nb, settings.size, settings.dim); // what is the best informed
	// position for each particle
	initMatrix(comm, settings.size, settings.size); // communications:who informs who
	// rows : those who inform
	// cols : those who are informed
	int improved; // whether solution->error was improved during
	// the last iteration

	int i, d, step;
	double a, b; // for matrix initialization
	double rho1, rho2; // random numbers (coefficients)
	double w; // current omega
	Inform_Fun inform_fun; // neighborhood update function
	double (*calc_inertia_fun)(int, pso_settings_t); // inertia weight update function
	
	std::default_random_engine generator(settings.seed);
	std::uniform_real_distribution<double> distribution(0.0,1.0);

	// SELECT APPROPRIATE NHOOD UPDATE FUNCTION
	switch (settings.nhood_strategy)
	{
	case PSO_NHOOD_GLOBAL :
		// comm matrix not used
		inform_fun = inform_global;
		break;
	case PSO_NHOOD_RING :
		init_comm_ring(comm, settings);
		inform_fun = inform_ring;
		break;
	default:
		init_comm_random(comm, settings);
		inform_fun = inform_random;
		break;
	}
	calc_inertia_fun = calc_inertia_lin_dec;
	// SELECT APPROPRIATE INERTIA WEIGHT UPDATE FUNCTION
	switch (settings.w_strategy)
	{
		/* case PSO_W_CONST : */
		/*     calc_inertia_fun = calc_inertia_const; */
		/*     break; */
	case PSO_W_LIN_DEC :
		calc_inertia_fun = calc_inertia_lin_dec;
		break;
	}

	// INITIALIZE SOLUTION
	solution->error = 0;//DBL_MAX;
	solution->stopping_criterion = false;
	improved = 0;

	// SWARM INITIALIZATION
	// for each particle
	int asd = int(sqrt(settings.size));
	double muc = double(asd-1);
	for (i=0; i<settings.size; i++) {
		// for each dimension
		int k = i / asd;
		int j = i % asd;
		for (d=0; d<settings.dim; d++) {
			pos[i][d] = pos_b[i][d] = settings.lo[d] + ((d==0?k:j) / muc) * (settings.hi[d] - settings.lo[d]);

			// generate two numbers within the specified range
			a = settings.lo[d] + (settings.hi[d] - settings.lo[d]) * \
				distribution(generator);
			b = settings.lo[d] + (settings.hi[d] - settings.lo[d]) *	\
				distribution(generator);
			// initialize position
			//pos[i][d] = a;
			// best position is the same
			//pos_b[i][d] = a;
			// initialize velocity
			vel[i][d] = (a-b) / 20.;
		}
		// update particle fitness
		fit[i] = obj_fun(pos[i], settings.dim, obj_fun_params);
		fit_b[i] = fit[i]; // this is also the personal best
		// update gbest??
		if (fit[i] > solution->error) {
			// update best fitness
			solution->error = fit[i];
			// copy particle pos to gbest vector
			std::copy ( pos[i].begin(), pos[i].end(), solution->gbest.begin() );
		}

	}

	// initialize omega using standard value
	w = PSO_INERTIA;
	// RUN ALGORITHM
	std::queue<double> errors;
	double preverr = 0;

	for (step=0; step<settings.steps; step++) {
		// update current step
		settings.step = step;
		// update inertia weight
		// do not bother with calling a calc_w_const function
		if (settings.w_strategy)
			w = calc_inertia_fun(step, settings);
		// check optimization goal
		if (solution->error <= settings.goal) {
			// SOLVED!!
			if (settings.print_every)
				printf("Goal achieved @ step %d (error=%.3e) :-)\n", step, solution->error);
			break;
		}

		if(errors.size() >= settings.stopping_criterion_window) {		
			double window = 0;
			for(std::queue<double> q(errors); !q.empty(); q.pop()) window += q.front();
			errors.pop();

			if(window < settings.stopping_criterion_treshold * settings.stopping_criterion_window) {
				solution->stopping_criterion = true;
				printf("Stopping criterion reached @ step %d (error=%.3e) :-)\n", step, solution->error);
					break;
			}
		}
		errors.push(abs(preverr-solution->error));
		preverr = solution->error;

		// update pos_nb matrix (find best of neighborhood for all particles)
		inform_fun(comm, pos_nb, pos_b, fit_b, solution->gbest,
			improved, settings);
		// the value of improved was just used; reset it
		improved = 0;

		// update all particles
		for (i=0; i<settings.size; i++) {
			// for each dimension
			for (d=0; d<settings.dim; d++) {
				// calculate stochastic coefficients
				rho1 = settings.c1 * distribution(generator);
				rho2 = settings.c2 * distribution(generator);
				// update velocity
				vel[i][d] = w * vel[i][d] +	\
					rho1 * (pos_b[i][d] - pos[i][d]) +	\
					rho2 * (pos_nb[i][d] - pos[i][d]);
				// update position
				pos[i][d] += vel[i][d];
				// clamp position within bounds?
				
				if (settings.clamp_pos) {
					if (pos[i][d] < settings.lo[d]) {
						pos[i][d] = settings.lo[d];
						vel[i][d] = 0;
					} else if (pos[i][d] > settings.hi[d]) {
						pos[i][d] = settings.hi[d];
						vel[i][d] = 0;
					}
				} else {
					// enforce periodic boundary conditions
					if (pos[i][d] < settings.lo[d]) {

						pos[i][d] = settings.hi[d] - fmod(settings.lo[d] - pos[i][d],
							settings.hi[d] - settings.lo[d]);
						vel[i][d] = 0;

						//assert(pos[i][d] > settings.lo[d] && pos[i][d] < settings.hi);

						/* printf("%f < x_lo=%.0f (v=%f) (mod=%f)\n",
						          pos[i][d], settings.lo, vel[i][d],
						          settings.hi - fmod(settings.lo - pos[i][d],
						   			     settings.hi - settings.lo));
						   assert(pos[i][d] > settings.lo && pos[i][d] < settings.hi); */

						//vel[i][d] = 0;
					} else if (pos[i][d] > settings.hi[d]) {

						pos[i][d] = settings.lo[d] + fmod(pos[i][d] - settings.hi[d],
							settings.hi[d] - settings.lo[d]);
						vel[i][d] = 0;

						/* printf("%f > x_hi=%.0f (v=%f) (mod=%f)\n",
						          pos[i][d], settings.hi, vel[i][d],
						          settings.lo + fmod(pos[i][d] - settings.hi,
						   			     settings.hi - settings.lo));
						   assert(pos[i][d] > settings.lo && pos[i][d] < settings.hi); */

					}
				}

			}

			// update particle fitness
			fit[i] = obj_fun(pos[i], settings.dim, obj_fun_params);
			// update personal best position?
			if (fit[i] > fit_b[i]) {
				fit_b[i] = fit[i];
				// copy contents of pos[i] to pos_b[i]
				std::copy(pos[i].begin(), pos[i].end(), pos_b[i].begin());
			}
			// update gbest??
			if (fit[i] > solution->error) {
				improved = 1;
				// update best fitness
				solution->error = fit[i];
				// copy particle pos to gbest vector
				std::copy ( pos[i].begin(), pos[i].end(), solution->gbest.begin() );
			}
		}

		if (settings.print_every && (step % settings.print_every == 0))
			printf("Step %d (w=%.2f) :: min err=%.5e\n", step, w, solution->error);

	}
}
