
#include <stdint.h>
#include <Rinternals.h>
#include <R_ext/Parse.h>

#include "rust/api.h"

static uintptr_t TAGGED_POINTER_MASK = (uintptr_t)1;

SEXP handle_result(SEXP res_) {
    uintptr_t res = (uintptr_t)res_;

    // An error is indicated by tag.
    if ((res & TAGGED_POINTER_MASK) == 1) {
        // Remove tag
        SEXP res_aligned = (SEXP)(res & ~TAGGED_POINTER_MASK);

        // Currently, there are two types of error cases:
        //
        //   1. Error from Rust code
        //   2. Error from R's C API, which is caught by R_UnwindProtect()
        //
        if (TYPEOF(res_aligned) == CHARSXP) {
            // In case 1, the result is an error message that can be passed to
            // Rf_errorcall() directly.
            Rf_errorcall(R_NilValue, "%s", CHAR(res_aligned));
        } else {
            // In case 2, the result is the token to restart the
            // cleanup process on R's side.
            R_ContinueUnwind(res_aligned);
        }
    }

    return (SEXP)res;
}

SEXP savvy_rust_balance_deviation__impl(SEXP c_arg__r_sample, SEXP c_arg__r_prob, SEXP c_arg__r_data) {
    SEXP res = savvy_rust_balance_deviation__ffi(c_arg__r_sample, c_arg__r_prob, c_arg__r_data);
    return handle_result(res);
}

SEXP savvy_rust_balanced__impl(SEXP c_arg__r_prob, SEXP c_arg__r_bal_data, SEXP c_arg__r_eps, SEXP c_arg__r_method) {
    SEXP res = savvy_rust_balanced__ffi(c_arg__r_prob, c_arg__r_bal_data, c_arg__r_eps, c_arg__r_method);
    return handle_result(res);
}

SEXP savvy_rust_balanced_stratified__impl(SEXP c_arg__r_prob, SEXP c_arg__r_bal_data, SEXP c_arg__r_strata, SEXP c_arg__r_eps, SEXP c_arg__r_method) {
    SEXP res = savvy_rust_balanced_stratified__ffi(c_arg__r_prob, c_arg__r_bal_data, c_arg__r_strata, c_arg__r_eps, c_arg__r_method);
    return handle_result(res);
}

SEXP savvy_rust_doubly_balanced__impl(SEXP c_arg__r_prob, SEXP c_arg__r_data, SEXP c_arg__r_bal_data, SEXP c_arg__r_eps, SEXP c_arg__r_bucket_size, SEXP c_arg__r_method) {
    SEXP res = savvy_rust_doubly_balanced__ffi(c_arg__r_prob, c_arg__r_data, c_arg__r_bal_data, c_arg__r_eps, c_arg__r_bucket_size, c_arg__r_method);
    return handle_result(res);
}

SEXP savvy_rust_doubly_balanced_stratified__impl(SEXP c_arg__r_prob, SEXP c_arg__r_data, SEXP c_arg__r_bal_data, SEXP c_arg__r_strata, SEXP c_arg__r_eps, SEXP c_arg__r_bucket_size, SEXP c_arg__r_method) {
    SEXP res = savvy_rust_doubly_balanced_stratified__ffi(c_arg__r_prob, c_arg__r_data, c_arg__r_bal_data, c_arg__r_strata, c_arg__r_eps, c_arg__r_bucket_size, c_arg__r_method);
    return handle_result(res);
}

SEXP savvy_rust_local_mean_variance__impl(SEXP c_arg__r_values, SEXP c_arg__r_prob, SEXP c_arg__r_data, SEXP c_arg__r_neighbours) {
    SEXP res = savvy_rust_local_mean_variance__ffi(c_arg__r_values, c_arg__r_prob, c_arg__r_data, c_arg__r_neighbours);
    return handle_result(res);
}

SEXP savvy_rust_pips_from_values__impl(SEXP c_arg__r_values, SEXP c_arg__r_sample_size) {
    SEXP res = savvy_rust_pips_from_values__ffi(c_arg__r_values, c_arg__r_sample_size);
    return handle_result(res);
}

SEXP savvy_rust_spatial_balance_measure__impl(SEXP c_arg__r_sample, SEXP c_arg__r_prob, SEXP c_arg__r_data, SEXP c_arg__r_method) {
    SEXP res = savvy_rust_spatial_balance_measure__ffi(c_arg__r_sample, c_arg__r_prob, c_arg__r_data, c_arg__r_method);
    return handle_result(res);
}

SEXP savvy_rust_spatially_balanced__impl(SEXP c_arg__r_prob, SEXP c_arg__r_data, SEXP c_arg__r_eps, SEXP c_arg__r_bucket_size, SEXP c_arg__r_method) {
    SEXP res = savvy_rust_spatially_balanced__ffi(c_arg__r_prob, c_arg__r_data, c_arg__r_eps, c_arg__r_bucket_size, c_arg__r_method);
    return handle_result(res);
}

SEXP savvy_rust_spatially_balanced_hierarchical__impl(SEXP c_arg__r_prob, SEXP c_arg__r_data, SEXP c_arg__r_sizes, SEXP c_arg__r_eps, SEXP c_arg__r_bucket_size, SEXP c_arg__r_method) {
    SEXP res = savvy_rust_spatially_balanced_hierarchical__ffi(c_arg__r_prob, c_arg__r_data, c_arg__r_sizes, c_arg__r_eps, c_arg__r_bucket_size, c_arg__r_method);
    return handle_result(res);
}

SEXP savvy_rust_unequal__impl(SEXP c_arg__r_prob, SEXP c_arg__r_eps, SEXP c_arg__r_method, SEXP c_arg__r_max_iter) {
    SEXP res = savvy_rust_unequal__ffi(c_arg__r_prob, c_arg__r_eps, c_arg__r_method, c_arg__r_max_iter);
    return handle_result(res);
}

SEXP savvy_rust_unequal_conditional_poisson__impl(SEXP c_arg__r_prob, SEXP c_arg__r_sample_size, SEXP c_arg__r_eps, SEXP c_arg__r_max_iter) {
    SEXP res = savvy_rust_unequal_conditional_poisson__ffi(c_arg__r_prob, c_arg__r_sample_size, c_arg__r_eps, c_arg__r_max_iter);
    return handle_result(res);
}


static const R_CallMethodDef CallEntries[] = {
    {"savvy_rust_balance_deviation__impl", (DL_FUNC) &savvy_rust_balance_deviation__impl, 3},
    {"savvy_rust_balanced__impl", (DL_FUNC) &savvy_rust_balanced__impl, 4},
    {"savvy_rust_balanced_stratified__impl", (DL_FUNC) &savvy_rust_balanced_stratified__impl, 5},
    {"savvy_rust_doubly_balanced__impl", (DL_FUNC) &savvy_rust_doubly_balanced__impl, 6},
    {"savvy_rust_doubly_balanced_stratified__impl", (DL_FUNC) &savvy_rust_doubly_balanced_stratified__impl, 7},
    {"savvy_rust_local_mean_variance__impl", (DL_FUNC) &savvy_rust_local_mean_variance__impl, 4},
    {"savvy_rust_pips_from_values__impl", (DL_FUNC) &savvy_rust_pips_from_values__impl, 2},
    {"savvy_rust_spatial_balance_measure__impl", (DL_FUNC) &savvy_rust_spatial_balance_measure__impl, 4},
    {"savvy_rust_spatially_balanced__impl", (DL_FUNC) &savvy_rust_spatially_balanced__impl, 5},
    {"savvy_rust_spatially_balanced_hierarchical__impl", (DL_FUNC) &savvy_rust_spatially_balanced_hierarchical__impl, 6},
    {"savvy_rust_unequal__impl", (DL_FUNC) &savvy_rust_unequal__impl, 4},
    {"savvy_rust_unequal_conditional_poisson__impl", (DL_FUNC) &savvy_rust_unequal_conditional_poisson__impl, 4},
    {NULL, NULL, 0}
};

void R_init_rsamplr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);

    // Functions for initialzation, if any.

}
