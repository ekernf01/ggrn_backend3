setwd("~/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/ggrn_backend3/hyperparameter_sweeps/")
library(ggplot2)
plot_performance = function(
  i, 
  metrics = c(
    "est_B_L1_norm",
    "true_G_L1_norm",
    "est_G_L1_norm",
    "est_G_L1_error",
    "true_G.S_L1_norm",
    "est_G.S_L1_norm",
    "est_G.S_L1_error",
    "B_L1_norm",
    "G_L1_norm"	,
    "G_L2_error",
    "num_epochs",
    "walltime"
  ),
  exclude = c("seed", "learning_rate"),
  factors_varied = NULL,
  factors_constant = NULL
){
  performance = read.csv(paste0(i, ".csv"), row.names = 1)
  metrics = intersect(metrics, names(performance))
  if(is.null(factors_varied)){
    factors_varied = names(performance)[sapply(performance, function(x) length(unique(x))>1)]
    factors_varied = setdiff(
      factors_varied, 
      metrics
    )
    factors_varied = setdiff(
      factors_varied, 
      exclude
    )
  }
  if(is.null(    factors_constant)){
    factors_constant = names(performance)[!sapply(performance, function(x) length(unique(x))>1)]
    factors_constant = setdiff(
      factors_constant, 
      metrics
    )    
    factors_constant = setdiff(
      factors_constant, 
      exclude
    )
  }
  factors_constant_string = ""
  for(f in factors_constant){
    factors_constant_string = paste0(factors_constant_string, f, ": ", unique(performance[[f]]), "\n")
  }
  for(y in metrics){
    x = factors_varied[1]
    if(length(factors_varied)==1){
      color = "seed"
      faceting_formula = NULL
    } 
    if(length(factors_varied)==2){
      color = factors_varied[2]
      faceting_formula = NULL
    } 
    if(length(factors_varied)==3){
      color = factors_varied[2]
      faceting_formula = paste0( " ~ ", factors_varied[3])
    } 
    if(length(factors_varied)==4){
      color = factors_varied[2]
      faceting_formula = paste0(factors_varied[4], " ~ ", factors_varied[3])
    } 
    performance[[x]] = factor(performance[[x]], levels = sort(unique(performance[[x]])))
    ggplot(performance) + 
      geom_point(mapping = aes_string(x=x, y=y, color=color)) + 
      annotate(geom = "text", 
               hjust = 0,
               label = factors_constant_string, 
               x = -3, 
               y = sqrt(max(performance[[y]], na.rm = T)*min(performance[[y]], na.rm = T))) +
      annotate(geom = "point", 
               hjust = 0,
               size=0,
               alpha=0,
               x = c(-3:-1),
               y = c(max(performance[[y]]), max(performance[[y]], na.rm = T), min(performance[[y]], na.rm = T)))  +
      ggtitle(paste0("Experiment ", i)) + 
      facet_grid(faceting_formula) + 
      scale_y_log10()
    dir.create(file.path(i), showWarnings = F)
    ggsave(paste0(file.path(i, y), ".pdf"), width = 7, height = 7)
  }
} 

for(i in 1:200){
  try(plot_performance(i), silent = T)
}

