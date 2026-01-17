use crate::config::*;
use crate::optimizer::lbf::LBFBuilder;
use crate::optimizer::separator::Separator;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use rand::{RngCore, SeedableRng};
use std::time::Duration;
use log::info;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::consts::LBF_SAMPLE_CONFIG;
use crate::optimizer::compress::compression_phase;
use crate::optimizer::explore::exploration_phase;
use crate::util::listener::{ReportType, SolutionListener};
use crate::util::terminator::Terminator;

pub mod lbf;
pub mod separator;
mod worker;
pub mod explore;
pub mod compress;

///Algorithm 11 from https://doi.org/10.48550/arXiv.2509.13329
pub fn optimize(
    instance: SPInstance,
    rng: Xoshiro256PlusPlus,
    sol_listener: &mut impl SolutionListener,
    terminator: &mut impl Terminator,
    expl_config: &ExplorationConfig,
    cmpr_config: &CompressionConfig,
    initial_solution: Option<&SPSolution>
) -> SPSolution {
    optimize_with_symmetric(instance, rng, sol_listener, terminator, expl_config, cmpr_config, initial_solution, false)
}

/// Optimize with optional symmetric mode.
/// In symmetric mode, items are placed only in the left half of the container,
/// and collision checking considers mirror positions.
pub fn optimize_with_symmetric(
    instance: SPInstance,
    mut rng: Xoshiro256PlusPlus,
    sol_listener: &mut impl SolutionListener,
    terminator: &mut impl Terminator,
    expl_config: &ExplorationConfig,
    cmpr_config: &CompressionConfig,
    initial_solution: Option<&SPSolution>,
    symmetric: bool,
) -> SPSolution {
    let mut next_rng = || Xoshiro256PlusPlus::seed_from_u64(rng.next_u64());
    let start_prob = match initial_solution {
        None => {
            let builder = LBFBuilder::new(instance.clone(), next_rng(), LBF_SAMPLE_CONFIG).construct();
            builder.prob
        }
        Some(init_sol) => {
            info!("[OPT] warm starting from provided initial solution");
            let mut prob = jagua_rs::probs::spp::entities::SPProblem::new(instance.clone());
            prob.restore(init_sol);
            prob
        }
    };

    // Calculate symmetric axis if in symmetric mode
    let symmetric_axis_x = if symmetric {
        Some(start_prob.strip_width() / 2.0)
    } else {
        None
    };

    if symmetric {
        info!("[OPT] symmetric mode enabled, axis at {:.3}", symmetric_axis_x.unwrap());
    }

    terminator.new_timeout(expl_config.time_limit);
    let mut expl_separator = Separator::new_with_symmetric(
        instance.clone(), start_prob, next_rng(), expl_config.separator_config, symmetric_axis_x
    );
    let solutions = exploration_phase(
        &instance,
        &mut expl_separator,
        sol_listener,
        terminator,
        expl_config,
    );
    let final_explore_sol = solutions.last().unwrap().clone();

    terminator.new_timeout(cmpr_config.time_limit);
    let mut cmpr_separator = Separator::new_with_symmetric(
        expl_separator.instance, expl_separator.prob, next_rng(), cmpr_config.separator_config, symmetric_axis_x
    );
    let cmpr_sol = compression_phase(
        &instance,
        &mut cmpr_separator,
        &final_explore_sol,
        sol_listener,
        terminator,
        cmpr_config,
    );

    sol_listener.report(ReportType::Final, &cmpr_sol, &instance);

    cmpr_sol
}