### jsonlite for parsing json
### agricolae for Tukey HSD
### xtable for latex format ANOVA table
pacman::p_load(agricolae, gplots, multcompView, ggplot2, jsonlite, xtable, stringr, stringi, gtools, RJSONIO)
### global vspace controls space between tables
vspace <- '-6.25mm'

get_data_for_n_factor_research_question <- function(file_name, factor_funcs,
                                                    factor_names){
### iterate through experiments in file_name
### decide factor from factor_func
### :param factor_name: goes in factor column of output dataframe
    
### parallel lists for dataframe to hold classifer and metrics
    factors <- matrix(nrow=0, ncol=length(factor_names))
    colnames(factors) <- factor_names
    auc <- c()
    f1 <- c()
    prec <- c()
    rec <- c()
    a_prc <- c()
    accuracy <- c()
    exp_names <- c()
    
### read file to nested list object
    document <- fromJSON(file_name, nullValue = NA)
    for (exp_name in names(document)){
### 10 iterations of 5-fold cross validation for
### every experiment
        cur_factors <- c()
        for (f in factor_funcs){
            cur_factors <- c(cur_factors, f(exp_name))
        }    
        for (exp in document[[exp_name]]$results_data){            
            for (iter in exp){
                factors <- rbind(factors, cur_factors)
                auc <- c(auc, iter$auc)
                f1 <- c(f1, iter$f1)
                prec <-c(prec, iter$precision)
                rec <- c(rec, iter$recall)
                a_prc <- c(a_prc, iter$a_prc)
                accuracy <- c(accuracy, iter$accuracy)
            }
        }
    }
    res_df <- data.frame('auc'=auc, 'f1'=f1, 'precision'=prec, 'recall'=rec,
                         'a_prc'=a_prc)
### accuracy was added later so not all results have it
    if (length(accuracy) > 0){
        res_df$accuracy = accuracy
        }
### "factors" will have list of factors for each experiment
### so need to split lists into columns
    for (factor_name in factor_names){
        res_df[[factor_name]] = factors[, factor_name]
    }
    return(res_df)
}

proper_form <- function(s){
### formats things properly for report
    if (s == 'auc'){
        return('\\ac{AUC}')
    } else if (s == 'a_prc'){
        return('\\ac{AUPRC}')
    } else {
        return(s)
    }
}

escape_underscores <- function(s){
### latex requires us to escape underscores
    return(gsub('_', ' ', s))
}

remove_vector_constructor <- function(s){
### the gsub, gsub, gsub is to work around
### the way that the aggregated hsd groups (agg)
### are stored as strings of vectors like "c("LightGBM", ...)
### that we do not know how to deal with
    return (gsub("\\)", "",
                 gsub("\"", "",
                      gsub("c\\(\"", "", escape_underscores(s)))))
}

write_hsd_table <- function(aov_clf_metric, agg, metric, label, caption){
### code common to one and two factor anova/hsd
### for writing tables of groups
### param aov_clf_metric: analysis of variance results
### param ac analysis: configuration object
### param agg aggregated: hsd test results
### param label - latex label for table
### para caption - latex caption for table
### return string of latex code for a table
    print(agg)
    end_row <- "\\\\ \\midrule\n"
    result <- "\\bgroup\n"
    result <- paste(result, "\\begin{table}[H]\n")
    result <- paste(result, "\\centering\n")
       result = paste0(result, "\t\\caption{", caption, "}\n")
    result <- paste(result, "\t\\begin{tabular}{l}\\toprule\n")
    i <- 1
    for (level in agg[[1]]){
        result = paste(result, "\tGroup", level, "consists of:",
                      remove_vector_constructor(agg[[2]][i]),
                       end_row)
        i <- i+1
    }
    result = paste(result, "\t\\end{tabular}\n")
    result = paste0(result, "\t\\label{tab:", label,  "}\n")
    result = paste(result, "\\vspace{", vspace, "}")
    result = paste(result, "\\end{table}")
    result = paste(result, "\\egroup")
    return(result)
}

englishify <- function(v){
### converts vector to english readable phrase
### for example c(1,2,3) -> 1, 2 and 3
### :param v: vector to be converted
    s <- stri_reverse(toString(v))
    s <- str_replace(s, ",", "dna ")
    return(stri_reverse(s))
}

make_level_dict <- function(agg) {
### make a dictionary of factor names
### to numbers for coloring HSD plots
    i <- 1
    group_numbers <- c()
    level_names <-c()
    group_labels <- c()
    for (group_label in agg[[1]]) {
        for (group_members in agg[[2]][i]) {
            group_numbers <- c(rep(i, length(group_members)), group_numbers)
            group_labels <- c(rep(group_label, length(group_members)), group_labels)
            level_names <- c(group_members, level_names)
            }
        i <- i + 1
    }
    result <- data.frame(group_numbers = group_numbers,
                         level_names = level_names,
                         group_labels = group_labels)
    ii <- order(result$group_labels)
    return(result[ii,])
}
end_anova_caption <- function(factors){
    print (paste('length of factors', length(factors)))
    if (length(factors) > 1){
        return("as factors of performance in terms of")
    } else {
        return ("as a factor of performance in terms of")
    }
}


n_factor_research_question <- function(exp_title, factors, metric, factor_funcs,
                                       input_file_name, output_file_name, make_boxplots = F,
                                       interactions = 1, add_hsd_groups = T,
                                       y_limits_vector=c(0, 1.1), hsd_alpha=0.01, format="json"){
### function to do ANOVA and HSD Test for n factors, no interaction
### :param question name: name of research question, usually Q1, Q2, etc.
### :param factors: list of treatment under anlysis
### :param metric: response analyized, for example area under precision recall curve
### :param factor_funcs: functions that returns level of factor, derived from experiment name
### :param format: 'json' or 'csv' experiment output file format
### in experiment results json file
### input_file_name: name of file containing experiment data, should be a file with a specific JSON
### format
### output_file_name: name of output latex file with experiment results
    write(
        paste("% statistical analysis of data from", input_file_name),
        file = output_file_name,
        append = T
    )

    write(
        paste0("\\subsection*{", exp_title, " Analysis of Results in Terms of ",
               proper_form(metric), "}\n"),
        file = output_file_name,
        append = T
    )
    
    if (format == 'json'){
        df <- get_data_for_n_factor_research_question(input_file_name, factor_funcs, factors)
    } else if (format == 'csv'){
        df <- read.csv(input_file_name)
    } else {
        stop('error unrecognized file format')
    }
        
    if (interactions > 1) {
        formula_str <- paste0("(", paste0(factors, collapse = " + ") ,")^", interactions)
    } else {
        formula_str <- paste(factors, collapse = " + ")
    }
    print(metric)
    factor_metric_model <- lm(paste(metric, "~", formula_str), data = df)
    aov_factor_metric <- aov(factor_metric_model, data=df)
    print(summary(aov_factor_metric))
    aov_table <-xtable(aov_factor_metric, caption = paste("ANOVA for", englishify(factors),
                               end_anova_caption(factors), proper_form(metric)
                               ),
                       label = 't' 
                       )
    aov_table_str <- print(aov_table, file = "/dev/null", caption.placement = "top")
    aov_table_str <- sub("\\[ht\\]", "[H]", aov_table_str)
    aov_table_str <- sub("\\{t\\}", paste0('{tab:anova-', metric,paste0(factors), exp_title, '}'), aov_table_str)
    aov_table_str <- sub("\\hline", "\\midrule", aov_table_str)
    aov_table_str <- sub("\\end\\{table\\}",
                         paste("\\vspace{", vspace, "}\n\\end{table}"), aov_table_str)
    write(aov_table_str, file = output_file_name, append = T)
    for (factor in factors){
  #      write(
  #          paste0("\n Analysis for the ", factor, " Factor\n\n"),
  #          file = output_file_name,
  #          append = T
  #  )

        hsd_res <- HSD.test(aov_factor_metric, factor, alpha=hsd_alpha, console=F, group=T)
### print hsd groups for treatments/factors
        agg <- aggregate(rownames(hsd_res$groups) ~ hsd_res$groups$groups, hsd_res$groups, paste)
### make dictionary of factors to numbers for colors
        hsd_factor_dict <- make_level_dict(agg)
        write(write_hsd_table(aov_clf_metric, agg, metric,
                              paste0('hsd-', factor, '-', metric, exp_title),
                              paste('HSD test groupings after ANOVA of',
                                    proper_form(metric), 'for the', factor, 'factor'
                                    )
                              ),
              file=output_file_name,
              append=T
              )
        if (make_boxplots == T) {
            plot_title <- paste( plotting_form(metric), "for levels of",
                                factor)
            png_file_name <- draw_boxplots(df[[metric]] ~ df[[factor]], df, factor, plotting_form(metric), 2,
                                           plot_title, hsd_factor_dict, output_file_name, add_hsd_groups, y_limits_vector)
            fig <- "\n\\begin{figure}[H]\n"
            fig <- paste0(fig, "\t\\caption{Boxplots of ", proper_form(metric),
                          " for levels of ", factor)
            if (add_hsd_groups){
                fig <- paste0(fig, "; HSD group printed above box}\n")
            } else {
                fig <- paste0(fig, "}\n")
            }
            fig <- paste0(fig, "\t\\includegraphics[width=\\linewidth]{", png_file_name, "}\n")
            fig <- paste0(fig, "\t\\label{fig:", png_file_name, "}\n")
            fig <- paste0(fig, "\\end{figure}\n")
            write(fig, file =  output_file_name, append = T)
        }
    }
    if (interactions > 1){
        print(paste('interactions', interactions))
        for(i in 2:interactions){
#### generate all i-combinations of factors
            combo_factors = combn(factors, i)
            print(paste('combo_factors', combo_factors))
            for (j in 1:dim(combo_factors)[2]){
### just check for interaction of all factors, only works for 2 case for now
                hsd_res <- HSD.test(aov_factor_metric, combo_factors[,j], alpha=0.01, console = F, group=T)
### print hsd groups for treatments/factors
                agg <- aggregate(
                    rownames(hsd_res$groups) ~ hsd_res$groups$groups, hsd_res$groups, paste)
### make dictionary of factors to numbers for colors
                hsd_factor_dict <- make_level_dict(agg)
                write(write_hsd_table(aov_clf_metric, agg, metric,
                                      paste0('hsd-', factor, '-', metric ),
                                      paste('HSD test groupings after ANOVA of',
                                            proper_form(metric), 'with interaction of',
                                            englishify(combo_factors[,j]),
                                            'as a factor'
                                            )
                                      ),
                      file=output_file_name,
                      append=T
                      )
            }
        }
    }
}


plotting_form <- function(s){
### convert metric name to something suitable for plotting
    if (s ==  "auc"){
        return("AUC")
    } else if(s == "a_prc"){
        return("AUPRC")
    } else {
        return(s)
    }
}


## Transparent colors
## Mark Gardener 2015
## https://www.dataanalytics.org.uk/make-transparent-colors-in-r/

t_col <- function(color, percent = 50, name = NULL) {
                                        #      color = color name
                                        #    percent = % transparency
                                        #       name = an optional name for the color

    ## Get RGB values for named color
    rgb.val <- col2rgb(color)

    ## Make new color using input color as base and alpha set by transparency
    t.col <- rgb(rgb.val[1], rgb.val[2], rgb.val[3],
                 max = 255,
                 alpha = percent * 255 / 100,
                 names = name)

    return(t.col)
}

draw_boxplots <- function(f, x, xlab, ylab, las, main, hsd_factor_dict, output_file_name, add_hsd_groups = T, y_limits_vector=c(0, 1.1)){
### generates box plots of experiment results per factor
### in the future we want  Tukey HSD groups on boxes
### return: name of file generated
    png_file_name <-   paste0(gsub("\\.", "-",
                                   gsub(":", "-",
                                        gsub('_','-',
                                             gsub(',','',
                                                  gsub(' ', '-',  paste0(output_file_name,
                                                                         tolower(main))))))),
                              ".png")
    # saves png file 
    png( png_file_name )
    par(mar=c(6, 4.1, 4.1, 2.1))
    levels <- as.character(unique(x[[xlab]]))
### need to sort levels in same way that x-axis labels are sorted
### by boxplot
    levels <- str_sort(levels, numeric = FALSE)
    n_colors <- length(unique(hsd_factor_dict$group_labels))
    color_pallet <- unlist(lapply(rainbow(n_colors), function(x) t_col(x, 10)))
    color_list <- c()
    boxplot_label_list <- c()
    for (level in levels) {
        color_list <- c(color_list, color_pallet[
            subset(hsd_factor_dict, level_names == level)$group_numbers])
        boxplot_label_list <-
            c(boxplot_label_list,
              as.vector(subset(hsd_factor_dict, level_names == level)$group_labels))
    }

    bxp <-boxplot(f, data = x, las = las, xlab = "", ylab = ylab, main = main, col = color_list,
                   lex.order=T,  ylim=y_limits_vector)

    # classifier names under boxes
    mtext(xlab, side = 1, line = 5)

### draw hsd group number over or on box plots
### so I don't have to keep re-typing the code
    if (add_hsd_groups) {
        positioning <- F
        if (positioning) {
            text(c(1 : length(hsd_factor_dict$level_names)),
                 bxp$stats[2, ] + 0.02,
                 boxplot_label_list)
        } else {
### Add hsd group on top
            text( 
                x=c(1:length(hsd_factor_dict$level_names)), 
                y=bxp$stats[nrow(bxp$stats),] + 0.05, 
                boxplot_label_list
            )
        }
    }
    dev.off()
    return(png_file_name)
}
