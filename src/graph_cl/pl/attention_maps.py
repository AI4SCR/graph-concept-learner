# TODO: save attetion map
# Save attention maps to file
# if gcl_cfg["aggregator"] == "transformer":
#     graph_concept_learner.aggregator.return_attention_map()
#     generate_and_save_attention_maps(
#         model=graph_concept_learner,
#         loader=DataLoader(
#             dataset=dataset,
#             batch_size=1,
#             follow_batch=follow_this,
#         ),
#         device=device,
#         follow_this_metrics=follow_this_metrics,
#         out_dir=out_dir,
#     )
#
# # End run
# mlflow.end_run()
