#pragma once
#define ZERO 1e-8				// Zero  Threshold
#define EPSILON 6e-4f			// Error Threshold
#define MAX_ITERATION 16		// Maximum Newton Iteration

//-------------------------------------------------------------------------------
// General Purpose Functions
//-------------------------------------------------------------------------------

__device__ bool		is_zero				(float n) { return fabs(n) < ZERO; }
__device__ bool		is_finite			(float n) { return (n <= FLT_MAX && n >= -FLT_MAX); }
__device__ bool		is_sign_different	(float a, float b) { return a < 0 != b < 0; }
__device__ float	sgn_b				(float value, float b) { return value * (b < 0 ? -1.0f : 1.0f); }
__device__ float	clamp				(float value, float min, float max) { return (value < min) ? min : (max < value) ? max : value; }

//-------------------------------------------------------------------------------
// High-Performance Polynomial Root Solver Functions
//-------------------------------------------------------------------------------

template <int N>	// Horner's Method is applied 
__device__ float	compute_polynomial(float const coeff[N + 1], float x) { float r = coeff[N]; for (int i = N - 1; i >= 0; --i) r = r * x + coeff[i]; return r; }

template <int N>
__device__ void		derivative(float deriv[N], float const coeff[N + 1]) { deriv[0] = coeff[1]; for (int i = 2; i <= N; ++i) deriv[i - 1] = i * coeff[i]; }

__device__ void		deflate(float deflated_polynomial[3], float const coeff[4], float root) { 
	deflated_polynomial[2] = coeff[3]; 
	for (int i = 2; i > 0; --i) deflated_polynomial[i - 1] = coeff[i] + deflated_polynomial[i] * root; 
}

template <int N>
__device__ float	find_single_root		(float const coeff[N + 1], float const deriv[N], float x_min, float x_max, float y_min, float y_max, float epsilon);

template <int N>
__device__ int		find_polynomial_roots	(float roots[N], float const coeff[N + 1], float x_init_min, float x_init_max, float epsilon = EPSILON);
__device__ int		find_cubic_roots		(float roots[3], float const coeff[4], float x_init_min, float x_init_max, float epsilon = EPSILON);
__device__ int		find_quadratic_roots	(float roots[2], float const coeff[3], float x_init_min, float x_init_max);

//-------------------------------------------------------------------------------
// Analytical Solution Functions
//-------------------------------------------------------------------------------

__device__ int		solve_quadratic (double coeff[3], double roots[2]);
__device__ int		solve_cubic		(double coeff[4], double roots[3]);
__device__ int		solve_quartic	(double coeff[5], double roots[4]);

//-------------------------------------------------------------------------------
// polynomial Class
//-------------------------------------------------------------------------------

template <int N>
class polynomial
{
public:
	// Coefficients of polynomial
	float			coeff[N + 1];

	//  Finds all roots of quartic polynomial between [x_init_min, x_init_max] and returns number of roots found
	__device__ int	find_roots(float roots[N], float x_init_min, float x_init_max, float epsilon = EPSILON) const { 
		return find_polynomial_roots <N>(roots, coeff, x_init_min, x_init_max, epsilon); 
	}
};

//-------------------------------------------------------------------------------
// Implementations
//-------------------------------------------------------------------------------

template <int N>
__device__ float find_single_root(float const coeff[N + 1], float const deriv[N], float x_init_min, float x_init_max, 
	float y_init_min, float y_init_max, float epsilon) {

	float x_r = (x_init_min + x_init_max) / 2;						// Initial guess at center of initial interval(2 critical points) 
																	// in order to prevent derivative approaches 0

	if (x_init_max - x_init_min <= 2 * epsilon) return x_r;			// Checks initial interval length for convergence criteria

	// Fast Numerical Root Finding for Cubic Polynomial
	if constexpr (N <= 3) {
		float x_r0 = x_r;											// Saves initial value

		// Newton iteration
		for (int i = 0; i < MAX_ITERATION; ++i) {
			float x_n = x_r - compute_polynomial<N>(coeff, x_r) / compute_polynomial<2>(deriv, x_r);

			x_n = clamp(x_n, x_init_min, x_init_max);				// Ensures convergence by containing next guess within interval

			if (abs(x_n - x_r) <= epsilon) return x_n;				// Until 'step size' is below given error (Termination condition)
			x_r = x_n;
		}
		if (!is_finite(x_r)) x_r = x_r0;							// Handles numerical truncation (ex: NaN)			
	}

	// Numerical Root Finding with Guaranteed Covergence
	float x_min = x_init_min;
	float x_max = x_init_max;
	float y_r = compute_polynomial<N>(coeff, x_r);		// f(x_r)

	while (true) {
		// Shorten root finding interval
		bool different = is_sign_different(y_init_min, y_r);

		if (different) x_max = x_r;
		else x_min = x_r;

		// Newton iteration
		float x_n = x_r - y_r / compute_polynomial<N - 1>(deriv, x_r);

		// Checks if Newton iteration fails
		if (x_n > x_min && x_n < x_max) {				// Sucess
			float stepsize = abs(x_n - x_r);
			x_r = x_n;

			// Termination condition
			if (stepsize > epsilon)		y_r = compute_polynomial<N>(coeff, x_r);
			else						break;
		}
		else {											// Fail
			// Bisection 
			x_r = (x_min + x_max) / 2;

			// Termination condition
			if (x_r == x_min || x_r == x_max || x_max - x_min <= 2 * epsilon)	break;
			y_r = compute_polynomial<N>(coeff, x_r);
		}
	}
	return x_r;
}

__device__ int find_quadratic_roots(float roots[2], float const coeff[3], float x_init_min, float x_init_max) {
	// Press' method (variant of quadratic formula)
	const float a = coeff[2];
	const float b = coeff[1];
	const float c = coeff[0];
	const float discriminant = b * b - 4 * a * c;

	// 2 real roots exits
	if (discriminant > 0) {
		const float temp = -0.5f * (b + sgn_b(sqrtf(discriminant), b));
		const float root_a = temp / a;
		const float root_b = c / temp;
		const float root_1 = fminf(root_a, root_b);
		const float root_2 = fmaxf(root_a, root_b);

		// Checks if each root in within interval
		int check_root_1 = (root_1 >= x_init_min) & (root_1 <= x_init_max);
		int check_root_2 = (root_2 >= x_init_min) & (root_2 <= x_init_max);
		roots[0] = root_1;
		roots[check_root_1] = root_2;
		return check_root_1 + check_root_2;
	}
	else if (discriminant < 0) return 0;	// No real root

	// 1 repeated real root exits
	const float root_1 = -0.5f * b / a;
	roots[0] = root_1;
	return (root_1 >= x_init_min) & (root_1 <= x_init_max);
}

__device__ int find_cubic_roots(float roots[3], float const coeff[4], float x_init_min, float x_init_max, float epsilon) {
	// Optimizing Cubic Root Finding via Deflation
	const float y_init_min = compute_polynomial<3>(coeff, x_init_min);
	const float y_init_max = compute_polynomial<3>(coeff, x_init_max);

	const float a = coeff[3] * 3;
	const float b = coeff[2] * 2;
	const float c = coeff[1];

	const float deriv[4] = { c, b, a, 0 };							// Coefficients of derivative (quadratic polynomial)
	const float discriminant = b * b - 4 * a * c;					// Delta from quadratic formula

	// 2 critical points exist
	if (discriminant > 0) {
		// Press' method (variant of quadratic formula)
		const float temp = -0.5f * (b + sgn_b(sqrtf(discriminant), b));
		float deriv_root_1 = temp / a;
		float deriv_root_2 = c / temp;
		const float x_1 = fminf(deriv_root_1, deriv_root_2);		// Zero-crossing of derivative
		const float x_2 = fmaxf(deriv_root_1, deriv_root_2);		// Zero-crossing of derivative

		if (is_sign_different(y_init_min, y_init_max)) {			// Only one root exits (lnterval contains only one monotonic piece)

			if (x_1 >= x_init_max || x_2 <= x_init_min || (x_1 <= x_init_min && x_2 >= x_init_max)) {
				roots[0] = find_single_root<3>(coeff, deriv, x_init_min, x_init_max, y_init_min, y_init_max, epsilon);
				return 1;
			}
		}
		else {														// No roots
			if ((x_1 >= x_init_max || x_2 <= x_init_min) || (x_1 <= x_init_min && x_2 >= x_init_max)) return 0;
		}

		// Interval Division
		int numRoots = 0;
		if (x_1 > x_init_min) {
			const float y_1 = compute_polynomial<3>(coeff, x_1);

			// A root exists in [x_init_min, x_1], first monotonic piece
			if (is_sign_different(y_init_min, y_1)) {
				roots[0] = find_single_root<3>(coeff, deriv, x_init_min, x_1, y_init_min, y_1, epsilon);

				// All 3 roots exist in [x_init_min, x_init_max]
				if (is_sign_different(y_1, y_init_max) || (x_2 < x_init_max && is_sign_different(y_1, compute_polynomial<3>(coeff, x_2)))) {

					// Deflation
					float deflated_polynomial[4];
					deflate(deflated_polynomial, coeff, roots[0]);
					return find_quadratic_roots(roots + 1, deflated_polynomial, x_1, x_init_max) + 1;
				}
				else return 1; 	// Only 1 root exist in [x_init_min, x_init_max]
			}
			if (x_2 < x_init_max) {
				const float y_2 = compute_polynomial<3>(coeff, x_2);

				// A root exists in [x_1, x_2], middle monotonic piece
				if (is_sign_different(y_1, y_2)) {
					roots[0] = find_single_root<3>(coeff, deriv, x_1, x_2, y_1, y_2, epsilon);

					// 2 roots exist in [x_init_min, x_init_max]
					if (is_sign_different(y_2, y_init_max)) {

						// Deflation
						float deflated_polynomial[4];
						deflate(deflated_polynomial, coeff, roots[0]);
						return find_quadratic_roots(roots + 1, deflated_polynomial, x_2, x_init_max) + 1;
					}
					else return 1;
				}

				// A root exists in [x_2, x_init_max], last monotonic piece
				if (is_sign_different(y_2, y_init_max)) {

					// Only 1 root exist in [x_init_min, x_init_max]
					roots[0] = find_single_root<3>(coeff, deriv, x_2, x_init_max, y_2, y_init_max, epsilon);
					return 1;
				}
			}
			else {
				// A root exists in [x_1, x_init_max], last monotonic piece
				if (is_sign_different(y_1, y_init_max)) {
					roots[0] = find_single_root<3>(coeff, deriv, x_1, x_init_max, y_1, y_init_max, epsilon);
					return 1;
				}
			}
		}
		else {
			const float y_2 = compute_polynomial<3>(coeff, x_2);

			// A root exists in [x_init_min, x_2], middle monotonic piece
			if (is_sign_different(y_init_min, y_2)) {
				roots[0] = find_single_root<3>(coeff, deriv, x_init_min, x_2, y_init_min, y_2, epsilon);

				// 2 roots exist in [x_init_min, x_init_max]
				if (is_sign_different(y_2, y_init_max)) {

					// Deflation
					float deflated_polynomials[4];
					deflate(deflated_polynomials, coeff, roots[0]);
					return find_quadratic_roots(roots + 1, deflated_polynomials, x_2, x_init_max) + 1;
				}
				else return 1;
			}

			// A root exists in [x_2, x_init_max], last monotonic piece
			if (is_sign_different(y_2, y_init_max)) {

				// Only 1 root exist in [x_init_min, x_init_max]
				roots[0] = find_single_root<3>(coeff, deriv, x_2, x_init_max, y_2, y_init_max, epsilon);
				return 1;
			}
		}
		return numRoots;

	}
	else {	// 1 repeated critical point or no critical points in real range
		if (is_sign_different(y_init_min, y_init_max)) {
			roots[0] = find_single_root<3>(coeff, deriv, x_init_min, x_init_max, y_init_min, y_init_max, epsilon);
			return 1;
		}
		return 0;
	}
}

template <int N>
__device__ int find_polynomial_roots(float roots[N], float const coeff[N + 1], float x_init_min, float x_init_max, float epsilon) {
	if		constexpr (N == 2)		return find_quadratic_roots(roots, coeff, x_init_min, x_init_max);
	else if constexpr (N == 3)		return find_cubic_roots(roots, coeff, x_init_min, x_init_max, epsilon);
	else if (coeff[N] == 0)			return find_polynomial_roots<N - 1>(roots, coeff, x_init_min, x_init_max, epsilon);	// When leading coefficient is 0
	else {																												// When quartic polynomial is given
		// Infinite loop can only happen when 2 consecutive guesses are on either side of inflection point, which is exactly at center of 2 critical points
		// To avoid it, Split interval between 2 critical points into intervals [x_1, x_c] and [x_c, x_2]
		float y_init_min = compute_polynomial<N>(coeff, x_init_min);
		float deriv[N];
		derivative<N>(deriv, coeff);

		float deriv_roots[N - 1];
		int num_critical_points = find_polynomial_roots<N - 1>(deriv_roots, deriv, x_init_min, x_init_max, epsilon);	// Computes critical points

		// Minimum setup
		float x[N + 1] = { x_init_min };
		float y[N + 1] = { y_init_min };

		// Interval devision
		for (int i = 0; i < num_critical_points; ++i) {
			x[i + 1] = deriv_roots[i];
			y[i + 1] = compute_polynomial<N>(coeff, deriv_roots[i]);
		}

		// Maximun setup
		x[num_critical_points + 1] = x_init_max;
		y[num_critical_points + 1] = compute_polynomial<N>(coeff, x_init_max);

		// Finds roots in each intervals
		int num_roots = 0;
		for (int i = 0; i <= num_critical_points; ++i) {
			// Considers only intervals containing roots
			if (is_sign_different(y[i], y[i + 1])) {
				roots[num_roots++] = find_single_root<N>(coeff, deriv, x[i], x[i + 1], y[i], y[i + 1], epsilon);
			}
		}
		return num_roots;
	}
}

//-------------------------------------------------------------------------------

__device__ int solve_quadratic(double coeff[3], double roots[2]) {
	double p, q, discriminant;

    // x^2 + px + q = 0 
    p = coeff[1] / (2 * coeff[2]);		// p = b/2a
    q = coeff[0] / coeff[2];			// q = c/a

    discriminant = p * p - q;			// b^2-4ac

    if (is_zero(discriminant)) {		// 1 repeated real roots
        roots[0] = -p;
        return 1;
    }
    else if (discriminant > 0) {		// 2 real roots 
		double sqrt_dis = sqrt(discriminant);

        roots[0] =  sqrt_dis - p;
        roots[1] = -sqrt_dis - p;
        return 2;	
    }
    else return 0;						// No real roots
}

__device__ int solve_cubic(double coeff[4], double roots[3]) {
    int     i, num_roots;
	double  sub;
	double  A, B, C;				
	double  squared_A, p, q;
	double  cubic_p, discriminant;

    // x^3 + Ax^2 + Bx + C = 0 
    A = coeff[2] / coeff[3];		// b/a		
    B = coeff[1] / coeff[3];		// c/a
    C = coeff[0] / coeff[3];		// d/a

    // Substitute x = y - A/3 to eliminate quadratic term
    // x^3 +px + q = 0 
    squared_A = A * A;
    p = 1.0 / 3 * (-1.0 / 3 * squared_A + B);
    q = 1.0 / 2 * (2.0 / 27 * A * squared_A - 1.0 / 3 * A * B + C);

    // Cardano's formula 
    cubic_p = p * p * p;
    discriminant = q * q + cubic_p;

    if (is_zero(discriminant)) {	
        if (is_zero(q)) {			// 1 triple root
            roots[0] = 0;
            num_roots = 1;
        }
        else {						// 1 single and 1 double roots
			double u = cbrt(-q);
            roots[0] = 2 * u;
            roots[1] = -u;
            num_roots = 2;
        }
    }
    else if (discriminant < 0) {	// 3 real roots (Casus irreducibilis)
		double phi = 1.0 / 3 * acos(-q / sqrt(-cubic_p));
		double temp = 2 * sqrt(-p);

        roots[0] =  temp * cos(phi);
        roots[1] = -temp * cos(phi + M_PI / 3);
        roots[2] = -temp * cos(phi - M_PI / 3);
        num_roots = 3;
    }
    else {							// 1 real root 
		double sqrt_dis = sqrt(discriminant);
		double u =  cbrt(sqrt_dis - q);
		double v = -cbrt(sqrt_dis + q);

        roots[0] = u + v;
        num_roots = 1;
    }

    // Resubstitute to compute x
    sub = 1.0 / 3 * A;
    for (i = 0; i < num_roots; ++i) roots[i] -= sub;

    return num_roots;
}

__device__ int solve_quartic(double coeff[5], double roots[4]) {
	int     i, num_roots;
	double  coeffs[4];
	double  z, u, v, sub;
	double  A, B, C, D;
	double  squared_A, p, q, r;

    // x^4 + Ax^3 + Bx^2 + Cx + D = 0 
	A = coeff[3] / coeff[4];			// b/a
    B = coeff[2] / coeff[4];			// c/a
    C = coeff[1] / coeff[4];			// d/a
    D = coeff[0] / coeff[4];			// e/a

    // Substitute x = y - A/4 to eliminate cubic term
    // x^4 + px^2 + qx + r = 0 
    squared_A = A * A;
    p = -3.0 / 8 * squared_A + B;
    q =  1.0 / 8 * squared_A * A - 1.0 / 2 * A * B + C;
    r = -3.0 / 256 * squared_A * squared_A + 1.0 / 16 * squared_A * B - 1.0 / 4 * A * C + D;

    if (is_zero(r)) {	 // No absolute term when r = 0, which means y(y^3 + py + q) = 0 
        coeffs[0] = q;
        coeffs[1] = p;
        coeffs[2] = 0;
        coeffs[3] = 1;

        num_roots = solve_cubic(coeffs, roots);

        roots[num_roots++] = 0;
    }
    else {				// r != 0, thus solves resolvent cubic equation

		// z^3 - 1/2pz^2 - rz + 1/2rp - 1/8q^2 = 0
        coeffs[0] = 1.0 / 2 * r * p - 1.0 / 8 * q * q;
        coeffs[1] = -r;
        coeffs[2] = -1.0 / 2 * p;
        coeffs[3] = 1;
        solve_cubic(coeffs, roots);		// Find 1 real root

		// Using found root z, divide into 2 quadratic equations
		// y^2 + uy + v = 0
		// y^2 - uy - v = 0
		// u = sqrt(z^2 - r)
		// v = sqrt(2z  - p)
        z = roots[0];
        u = z * z - r;
        v = 2 * z - p;

		// Prevents sqrt(negative value)
        if (is_zero(u))		u = 0;			
        else if (u > 0)		u = sqrt(u);
        else  return 0;

        if (is_zero(v))		v = 0;
        else if (v > 0)		v = sqrt(v);
        else  return 0;

		// First quadratic equation
        coeffs[0] = z - u;
        coeffs[1] = q < 0 ? -v : v;
        coeffs[2] = 1;
        num_roots = solve_quadratic(coeffs, roots);

		// Second quadratic equation
        coeffs[0] = z + u;
        coeffs[1] = q < 0 ? v : -v;
        coeffs[2] = 1;
        num_roots += solve_quadratic(coeffs, roots + num_roots);
    }

    // Resubstitute to compute x
    sub = 1.0 / 4 * A;
    for (i = 0; i < num_roots; ++i) roots[i] -= sub;

    return num_roots;
}