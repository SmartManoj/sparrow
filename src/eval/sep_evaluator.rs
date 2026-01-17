use jagua_rs::collision_detection::hazards::collector::HazardCollector;
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::eval::specialized_jaguars_pipeline::{collect_poly_collisions_in_detector_custom, SpecializedHazardCollector};
use crate::quantify::tracker::CollisionTracker;
use crate::symmetric::mirror_transformation;
use jagua_rs::entities::Item;
use jagua_rs::entities::Layout;
use jagua_rs::entities::PItemKey;
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::primitives::SPolygon;
use jagua_rs::geometry::geo_traits::TransformableFrom;

pub struct SeparationEvaluator<'a> {
    layout: &'a Layout,
    item: &'a Item,
    collector: SpecializedHazardCollector<'a>,
    shape_buff: SPolygon,
    mirror_shape_buff: SPolygon,
    n_evals: usize,
    symmetric_axis_x: Option<f32>,
}

impl<'a> SeparationEvaluator<'a> {
    pub fn new(
        layout: &'a Layout,
        item: &'a Item,
        current_pk: PItemKey,
        ct: &'a CollisionTracker,
    ) -> Self {
        Self::new_with_symmetric(layout, item, current_pk, ct, None)
    }

    pub fn new_with_symmetric(
        layout: &'a Layout,
        item: &'a Item,
        current_pk: PItemKey,
        ct: &'a CollisionTracker,
        symmetric_axis_x: Option<f32>,
    ) -> Self {
        let collector = SpecializedHazardCollector::new(layout, ct, current_pk);

        Self {
            layout,
            item,
            collector,
            shape_buff: item.shape_cd.as_ref().clone(),
            mirror_shape_buff: item.shape_cd.as_ref().clone(),
            n_evals: 0,
            symmetric_axis_x,
        }
    }
}

impl<'a> SampleEvaluator for SeparationEvaluator<'a> {
    /// Evaluates a transformation. An upper bound can be provided to early terminate the process.
    /// Algorithm 7 from https://doi.org/10.48550/arXiv.2509.13329
    /// In symmetric mode, also checks for collisions at the mirror position.
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        self.n_evals += 1;
        let cde = self.layout.cde();

        //samples which evaluate higher than this will certainly be rejected
        let loss_bound = match upper_bound {
            Some(SampleEval::Collision { loss }) => loss,
            Some(SampleEval::Clear { .. }) => 0.0,
            _ => f32::INFINITY,
        };
        //reload the hazard collector to prepare for a new query
        self.collector.reload(loss_bound);

        //query the CDE, all colliding hazards will be stored in the detection map
        collect_poly_collisions_in_detector_custom(cde, &dt, &mut self.shape_buff, self.item.shape_cd.as_ref(), &mut self.collector);

        let original_result = if self.collector.early_terminate(&self.shape_buff) {
            SampleEval::Invalid
        } else if self.collector.is_empty() {
            SampleEval::Clear { loss: 0.0 }
        } else {
            SampleEval::Collision {
                loss: self.collector.loss(&self.shape_buff),
            }
        };

        // In symmetric mode, also check the mirror position
        if let Some(axis_x) = self.symmetric_axis_x {
            match original_result {
                SampleEval::Invalid => SampleEval::Invalid,
                SampleEval::Clear { loss: orig_loss } | SampleEval::Collision { loss: orig_loss } => {
                    // Compute mirror transformation
                    let mirror_dt = mirror_transformation(dt, axis_x);

                    // Reload collector for mirror check
                    let mirror_loss_bound = loss_bound - orig_loss;
                    if mirror_loss_bound <= 0.0 {
                        return SampleEval::Collision { loss: orig_loss };
                    }
                    self.collector.reload(mirror_loss_bound);

                    // Check collisions at mirror position
                    collect_poly_collisions_in_detector_custom(
                        cde,
                        &mirror_dt,
                        &mut self.mirror_shape_buff,
                        self.item.shape_cd.as_ref(),
                        &mut self.collector
                    );

                    if self.collector.early_terminate(&self.mirror_shape_buff) {
                        SampleEval::Invalid
                    } else if self.collector.is_empty() {
                        if orig_loss == 0.0 {
                            SampleEval::Clear { loss: 0.0 }
                        } else {
                            SampleEval::Collision { loss: orig_loss }
                        }
                    } else {
                        let mirror_loss = self.collector.loss(&self.mirror_shape_buff);
                        SampleEval::Collision {
                            loss: orig_loss + mirror_loss,
                        }
                    }
                }
            }
        } else {
            original_result
        }
    }

    fn n_evals(&self) -> usize {
        self.n_evals
    }
}

