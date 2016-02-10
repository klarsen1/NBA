library(xlsx)
library(dplyr)
library(ggplot2)
library(readxl)
library(data.table)
library(scales)
library(randomForestSRC)
library(doMC)
library(glmnet)
library(gridExtra)
library(grid)
library(tidyr)

multiplot <- function(..., plotlist=NULL, cols=1, layout=NULL) {
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

setwd("/Users/kimlarsen/Code/analytics/analysis/nba/predict")

source("auc.R")

### Global settings
cutoff <- 8 # minutes per game. if a player plays less than this amount, he is excluded
nclus <- 25 # number of archetypes
ntrees <- 100 # number trees in the RF solution
window <- 91
include_winperc <- FALSE
method <- "EN"
prior <- FALSE
alpha <- 0 # for elastic net


### Read the raw data
setwd("/Users/kimlarsen/Code/analytics/analysis/nba/data")
box_scores <- readRDS("BOX_SCORES.RDA")
game_scores <- readRDS("GAME_SCORES.RDA")

#playoff_date <- max(subset(box_scores, season==2014 & playoffs==0)$DATE)
playoff_date <- as.Date("2016-05-01")

setwd("/Users/kimlarsen/Code/analytics/analysis/nba/predict")

### Get means for centroids
means <- box_scores %>%
         group_by(PLAYER_FULL_NAME) %>%
         filter(season<2015 & playoffs==0) %>%
         summarise(assists=mean(assists),
                   offensive_rebounds=mean(offensive_rebounds),
                   defensive_rebounds=mean(defensive_rebounds),
                   turnovers=mean(turnovers),
                   threepointers_made=mean(threepointers_made), 
                   steals=mean(steals),
                   points=mean(points),
                   minutes=mean(minutes),
                   threepoint_attempts=mean(threepoint_attempts), 
                   fieldgoal_attempts=mean(fieldgoal_attempts), 
                   fieldgoals_made=mean(fieldgoals_made),
                   freethrows_made=mean(freethrows_made),
                   freethrow_attempts=mean(freethrow_attempts), 
                   blocks=mean(blocks),
                   share_of_minutes=mean(share_of_minutes)) %>%
              mutate(fieldgoal_percent=ifelse(fieldgoal_attempts>0, fieldgoals_made/fieldgoal_attempts, 0), 
                     freethrow_percent=ifelse(freethrow_attempts>0, freethrows_made/freethrow_attempts, 0), 
                     efficiency=(blocks + points + offensive_rebounds + defensive_rebounds + assists + steals - (fieldgoal_attempts - fieldgoals_made) - (freethrow_attempts - freethrows_made) - turnovers)) %>%
              replace(is.na(.), 0) %>%
              select(-threepoint_attempts, -freethrow_attempts, -fieldgoal_attempts, -points)

setwd("/Users/kimlarsen/Google Drive/NBA/cluster_analysis")

### Check quantiles for excluding players who play very little
quantile(means$minutes,  probs = c(0.1, 0.5, 1, 2, 5, 10, 50, 100)/100)


### Find the ideal number of clusters
means_no_scrubs <- subset(means, minutes>cutoff)

length(unique(means_no_scrubs$PLAYER_FULL_NAME))

means_no_scrubs$minutes <- NULL

standardized <- scale(means_no_scrubs[,sapply(means_no_scrubs, is.numeric)])
wss <- data.frame(matrix(nrow=50, ncol=2))
names(wss) <- c("clusters", "prop_wss")
for (i in 1:75){ 
  wss[i,"clusters"] <- i
  if (i==1){
    wss[i, "prop_wss"] <- 0
    set.seed(2015)
    v <- sum(kmeans(standardized, centers=i, nstart=25)$withinss)
  } else{
    wss[i,"prop_wss"] <- 1-sum(kmeans(standardized, centers=i, nstart=25)$withinss)/v
  }
}

ggplot(data=wss, aes(x=clusters, y=prop_wss)) + geom_line() + scale_y_continuous(breaks=pretty_breaks(n=10))


#### Get the final centroids

set.seed(2015)
km <- kmeans(standardized, centers=nclus, nstart=25)
centroids <- km$centers

#### Save the cluster solution to a CSV
production_centroids <- data.frame(cbind(means_no_scrubs, km$cluster), stringsAsFactors = FALSE) %>%
                        rename(Cluster = km.cluster) %>%
                        arrange(Cluster, PLAYER_FULL_NAME)


### Helper function to match players against teh closest centroid
closest.cluster <- function(x) {
  cluster.dist <- apply(centroids, 1, function(y) sqrt(sum((x-y)^2)))
  return(which.min(cluster.dist)[1])
}


### Now get the moving forecasts
  #box_scores <- subset(box_scores, playoffs==0)
  dates <- sort(unique(box_scores$DATE))
  datemap <- cbind.data.frame(dates, row_number(dates))
  names(datemap) <- c("DATE", "ROW")
  start_date <- min(subset(box_scores, season==2015)$DATE)
  start_index <- subset(datemap, DATE==start_date)$ROW
  scores <- list()
  importances <- list()
  team_stats <- list()
  cluster_movement <- list()
  surpluses <- list()
  counter <- 1
  for (i in start_index:length(dates)){
    thisdata <- dplyr::filter(box_scores, DATE<dates[i] & DATE>dates[i-window]) 
    means <- thisdata %>% 
      group_by(PLAYER_FULL_NAME) %>%
      summarise(assists=mean(assists),
                offensive_rebounds=mean(offensive_rebounds),
                defensive_rebounds=mean(defensive_rebounds),
                turnovers=mean(turnovers),
                threepointers_made=mean(threepointers_made), 
                steals=mean(steals),
                points=mean(points),
                minutes=mean(minutes),
                threepoint_attempts=mean(threepoint_attempts), 
                fieldgoal_attempts=mean(fieldgoal_attempts), 
                fieldgoals_made=mean(fieldgoals_made),
                freethrows_made=mean(freethrows_made),
                freethrow_attempts=mean(freethrow_attempts), 
                blocks=mean(blocks),
                share_of_minutes=mean(share_of_minutes)) %>%
      mutate(fieldgoal_percent=ifelse(fieldgoal_attempts>0, fieldgoals_made/fieldgoal_attempts, 0), 
             freethrow_percent=ifelse(freethrow_attempts>0, freethrows_made/freethrow_attempts, 0), 
             efficiency=(blocks + points + offensive_rebounds + defensive_rebounds + assists + steals - (fieldgoal_attempts - fieldgoals_made) - (freethrow_attempts - freethrows_made) - turnovers)) %>%
      replace(is.na(.), 0) %>%
      select(-threepoint_attempts, -freethrow_attempts, -fieldgoal_attempts, -points)
    
    avg_minutes <- select(means, PLAYER_FULL_NAME, minutes)
 
    scrubs <- filter(means, minutes<=cutoff) %>% mutate(Cluster=0) %>% select(PLAYER_FULL_NAME, Cluster)
    means_no_scrubs <- filter(means, minutes>cutoff) %>% select(-PLAYER_FULL_NAME, -minutes)
    names <- subset(means, minutes>cutoff)$PLAYER_FULL_NAME
    
    scaled <- scale(means_no_scrubs)
    
    clusters <- data.frame(cbind(apply(scaled, 1, closest.cluster), names), stringsAsFactors = FALSE)
    names(clusters) <- c("Cluster", "PLAYER_FULL_NAME")
    clusters$Cluster <- as.numeric(clusters$Cluster)
    clusters <- rbind.data.frame(clusters, scrubs)
    clusters <- inner_join(clusters, avg_minutes, by="PLAYER_FULL_NAME")
    active <- filter(box_scores, DATE==dates[i]) %>% select(PLAYER_FULL_NAME)
    clusters <- inner_join(clusters, active, by="PLAYER_FULL_NAME")

    ### Find a team for each player
    team_members <- thisdata %>% group_by(PLAYER_FULL_NAME, selected_team) %>%
                    summarise(minutes=sum(minutes)) %>%
                    mutate(rank=rank(-minutes)) %>%
                    filter(rank==1) %>%
                    select(PLAYER_FULL_NAME, selected_team) %>%
                    distinct() %>%
                    ungroup()

    clusters <- inner_join(clusters, team_members, by="PLAYER_FULL_NAME") 
    
    cluster_movement[[counter]] <- clusters
    cluster_movement[[counter]]$DATE <- dates[i]
    
    ### Get weighted win percentages for the selected team
    win_perc_opposing <- thisdata %>% group_by(opposing_team) %>% 
      summarise(win_opposing=mean(as.numeric(selected_team_win==0))) %>%
      ungroup()

    win_perc1 <- thisdata %>% group_by(selected_team) %>% 
                         inner_join(win_perc_opposing, by="opposing_team") %>%
                         mutate(w=selected_team_win*win_opposing) %>%
                         summarise(n=sum(w), d=sum(win_opposing)) %>%
                         mutate(weighted_win_perc1=ifelse(d>0, n/d, 0)) %>%
                         replace(is.na(.), 0) %>%
                         select(weighted_win_perc1, selected_team) %>%
                         ungroup()
    
    rm(win_perc_opposing)

    ### Get weighted win percentages for the opposing team
    win_perc_opposing <- thisdata %>% group_by(selected_team) %>% 
      summarise(win_opposing=mean(selected_team_win)) %>%
      ungroup() %>%
      rename(opposing_team=selected_team)
    
    win_perc2 <- thisdata %>% group_by(opposing_team) %>% 
      inner_join(win_perc_opposing, by="opposing_team") %>%
      mutate(w=as.numeric(selected_team_win==0)*win_opposing) %>%
      summarise(n=sum(w), d=sum(win_opposing)) %>%
      mutate(weighted_win_perc2=ifelse(d>0, n/d, 0)) %>%
      replace(is.na(.), 0) %>%
      select(weighted_win_perc2, opposing_team) %>%
      ungroup() %>%
      rename(selected_team=opposing_team)

    win_perc <- inner_join(win_perc1, win_perc2, by="selected_team") %>%
                mutate(win_perc_delta = weighted_win_perc1 - weighted_win_perc2) %>%
                select(selected_team, win_perc_delta)
    
    rfdata <- inner_join(thisdata, select(clusters, PLAYER_FULL_NAME, Cluster), by="PLAYER_FULL_NAME")
    thisday <- inner_join(filter(box_scores, DATE==dates[i]), select(clusters, PLAYER_FULL_NAME, Cluster), by="PLAYER_FULL_NAME")

    for (j in 0:nclus){
      rfdata[,paste0("share_minutes_cluster_", j)] <- rfdata$share_of_minutes_signed * as.numeric(rfdata$Cluster==j)
      thisday[,paste0("share_minutes_cluster_", j)] <- thisday$share_of_minutes_signed * as.numeric(thisday$Cluster==j)
    }

    rfdata <- dplyr::select(rfdata, selected_team, selected_team_win, starts_with("share_minutes_cluster_"), home_team_selected, game_id) %>%
      group_by(game_id, selected_team) %>%
      summarise_each(funs(sum)) %>%
      mutate(selected_team_win=as.numeric(selected_team_win>0), 
             home_team_selected=as.numeric(home_team_selected>0)) %>%
      ungroup() %>% 
      left_join(win_perc, by="selected_team") %>%
      replace(is.na(.), 0)

    ### write out records for plotting
    for_bar_plot <- dplyr::select(thisday, DATE, selected_team, opposing_team, starts_with("share_minutes_cluster_")) %>%
      group_by(DATE, selected_team, opposing_team) %>%
      summarise_each(funs(sum)) %>%
      ungroup() %>%
      gather(cluster, surplus, share_minutes_cluster_0:share_minutes_cluster_25) %>%
      mutate(cluster = as.numeric(substr(cluster, 23, 24))) %>%
      arrange(DATE, selected_team, opposing_team, cluster)
    
    thisday <- dplyr::select(thisday, selected_team, selected_team_win, starts_with("share_minutes_cluster_"), home_team_selected, game_id) %>%
      group_by(game_id, selected_team) %>%
      summarise_each(funs(sum)) %>%
      mutate(selected_team_win=as.numeric(selected_team_win>0), 
             home_team_selected=as.numeric(home_team_selected>0)) %>%
      ungroup() %>%
      left_join(win_perc, by="selected_team") %>%
      replace(is.na(.), 0)
    
    
    if (include_winperc==FALSE){
      rfdata$win_perc_delta <- NULL
      thisday$win_perc_delta <- NULL
    }
    
    if (method=="RF"){
       rfdata$Y <- as.factor(rfdata$selected_team_win)
       rf.grow <- rfsrc(Y ~ ., data=data.frame(dplyr::select(rfdata, -selected_team_win, -game_id, -selected_team)), ntree=ntrees, seed=2015)
       prob_win <- predict(rf.grow, newdata=data.frame(dplyr::select(thisday, -selected_team_win, -game_id, -selected_team)), outcome="train")    
       importance <- data.frame(rf.grow$importance)
       importance$variable <- row.names(importance)
       importance <- arrange(importance, X1) %>% select(variable, X1) %>% rename(strength=X1) %>% mutate(DATE=dates[i])
       importances[[counter]] <- importance
       result <-  data.frame(cbind(select(thisday, game_id, selected_team_win), prob_win$predicted[,2]), stringsAsFactors = FALSE)
    } else{
      f <- as.formula(Y ~ .)
      Y <- rfdata$selected_team_win
      X <- model.matrix(f, data.frame(dplyr::select(rfdata, -selected_team_win, -game_id, -selected_team)))[, -1]
      model <- cv.glmnet(y=Y, x=X, family="binomial", alpha=alpha, parallel=FALSE, nfolds=10)   
      c <- as.matrix(coef(model, s=model$lambda.1se))
      selected <- cbind.data.frame(sapply(row.names(c), as.character), sapply(c, as.numeric))
      names(selected) <- c("Variable", "Coeff")
      selected <- subset(selected, Variable != "(Intercept)") %>% mutate(DATE=dates[i])
      cvm <- cbind.data.frame(model$lambda, model$cvm)
      names(cvm) <- c("LAMBDA", "CVM")
      rm(Y)
      
      surpluses[[counter]] <- for_bar_plot

      ## apply prior
      p <- data.frame(cbind(1/(1+exp(-predict(model, newx=X, s=model$lambda.1se))), dplyr::select(rfdata, selected_team, selected_team_win)), stringsAsFactors=FALSE)
      names(p) <- c("prob", "selected_team", "win")
      p$prob <- as.numeric(p$prob)
      
      a <- p %>% group_by(selected_team) %>%
           summarise(posterior=mean(prob),
                     prior=mean(win)) %>%
           mutate(adj = log( (1-prior)*posterior / (prior*(1-posterior)))) %>%
           dplyr::select(selected_team, adj) %>%
           ungroup()

      if (prior==FALSE){
         a$adj <- 0 ## reomove prior
      }
      
      thisday <- left_join(thisday, a, by="selected_team") %>%
                 replace(is.na(.), 0)
      rm(X)
      
      Y <- thisday$selected_team_win
      #thisday$int <- thisday$home_team_selected * thisday$win_perc_delta
      X <- model.matrix(f, data.frame(dplyr::select(thisday, -selected_team_win, -game_id, -selected_team, -adj)))[, -1]
      prob_win <- 1/(1+exp(-X%*%c[-1] + thisday$adj))
      importances[[counter]] <- arrange(selected, Coeff)
      rm(selected)
      result <-  data.frame(cbind(select(thisday, game_id, selected_team_win), prob_win), stringsAsFactors = FALSE)
    }
    names(result) <- c("game_id", "selected_team_win", "prob_win_selected_team")
    result$pred_win <- as.numeric(result$prob_win_selected_team>0.5)
    result <- inner_join(result, select(game_scores, -selected_team_win), by="game_id")
    scores[[counter]] <- result
    
    team_mix <- select(rfdata, game_id, starts_with("share_minutes_cluster_")) %>% 
                inner_join(thisdata, by=c("game_id")) %>%
                select(selected_team, opposing_team, starts_with("share_minutes_cluster_")) %>%
                group_by(selected_team, opposing_team) %>%
                summarise_each(funs(mean)) %>% 
                rename(team=selected_team) %>%
                mutate(DATE=dates[i]) %>%
                ungroup()
    
    team_stats[[counter]] <- team_mix
    
    print(paste0("Clusters= ", nclus))
    print(paste0("N = ", nrow(rfdata)))
    print(paste0("Date = ", dates[i]))
    print(paste0("Players = ", nrow(means_no_scrubs)))

    rm(thisday)
    rm(scaled)
    rm(clusters)
    rm(avg_minutes)
    rm(result)
    rm(team_mix)
    rm(means_no_scrubs)
    rm(scrubs)
    rm(rfdata)
    rm(thisdata)
    rm(win_perc_opposing)
    rm(win_perc)
    rm(win_perc1)
    rm(win_perc2)
    rm(team_members)
    rm(active)
    rm(for_bar_plot)
    
    counter <- counter + 1
    
  }

### End of scoring code  
    
scores <- data.frame(rbindlist(scores), stringsAsFactors = FALSE)
importances <- data.frame(rbindlist(importances), stringsAsFactors = FALSE) %>% mutate(playoffs=as.numeric(DATE>playoff_date))
team_stats <- data.frame(rbindlist(team_stats), stringsAsFactors = FALSE) %>% 
              arrange(team, DATE) %>% mutate(playoffs=as.numeric(DATE>playoff_date))
cluster_movement <- data.frame(rbindlist(cluster_movement), stringsAsFactors = FALSE) %>% 
                    arrange(PLAYER_FULL_NAME, DATE) %>% mutate(playoffs=as.numeric(DATE>playoff_date))
surpluses <- data.frame(rbindlist(surpluses), stringsAsFactors = FALSE)

### Get the C stat

paste0("C: ", AUC(scores$selected_team_win, scores$prob_win_selected_team)[1])

d <- subset(scores, DATE>=as.Date("2015-10-27"))
paste0("Accuracy: ", sum((d$selected_team_win==d$pred_win))/nrow(d))

View(subset(scores, selected_team %in% c("Golden State", "Cleveland") & opposing_team %in% c("Golden State", "Cleveland")))
     
View(subset(scores, selected_team %in% c("Golden State", "Oklahoma City") & opposing_team %in% c("Golden State", "Oklahoma City")))

View(subset(scores, selected_team %in% c("Golden State", "LA Clippers") & opposing_team %in% c("Golden State", "LA Clippers")))

### Summarize team performances -- have to do this twice because the data was collapsed into matchups
summary1 <- scores %>%
            group_by(playoffs, selected_team) %>%
  mutate(match=as.numeric(selected_team_win==pred_win)) %>%
  summarise(pred_win=sum(pred_win),
            prob_win_selected_team=sum(prob_win_selected_team),
            win=sum(selected_team_win),
            match=sum(match),
            games=n()) %>%
            rename(team=selected_team) %>%
  ungroup()

summary2 <- scores %>%
  group_by(playoffs, opposing_team) %>%
  mutate(match = as.numeric((selected_team_win==0)==(pred_win==0))) %>%
  summarise(pred_win=sum(as.numeric(pred_win==0)),
            prob_win_selected_team=sum(1-prob_win_selected_team),
            win=sum(selected_team_win==0),
            match=sum(match),
            games=n()) %>%
  rename(team=opposing_team) %>%
  ungroup()

summary <- rbind.data.frame(summary1, summary2) %>% group_by(playoffs, team) %>%
           summarise(win_rate=sum(win), 
                     prob_win_selected_team=sum(prob_win_selected_team),
                     games=sum(games), 
                     pred_win=sum(pred_win), 
                     match=sum(match)) %>%
           mutate(actual_win_rate = win_rate/games, 
                 prob_win_selected_team = prob_win_selected_team/games, 
                 pred_win_rate = pred_win / games,
                 accuracy = match / games, 
                 lift_over_coin_flip = accuracy/0.5-1, 
                 actual_wins=win_rate, 
                 predicted_wins=pred_win) %>%
           arrange(playoffs, pred_win_rate) %>%
           select(-match, -prob_win_selected_team, -win_rate, -pred_win) %>%
           mutate(pred_rank=rank(-pred_win_rate), 
                  actual_rank=rank(-actual_win_rate)) %>%
           ungroup()

View(summary)

setwd("/Users/kimlarsen/Code/analytics/analysis/nba/results")

saveRDS(importances, "importance_history.RDA")
saveRDS(cluster_movement, "cluster_history.RDA")
saveRDS(scores, "scores_and_predictions.RDA")
saveRDS(team_stats, "historical_team_cluster_deltas.RDA")
saveRDS(summary, "high_level_summary.RDA")
saveRDS(production_centroids, "centroids_from_2013.RDA")

setwd("/Users/kimlarsen/Code/analytics/analysis/nba/results")

importances <- readRDS("importance_history.RDA")
cluster_movement <- readRDS("cluster_history.RDA")
scores <- readRDS("scores_and_predictions.RDA")
team_stats <- readRDS("historical_team_cluster_deltas.RDA")
summary <- readRDS("high_level_summary.RDA")
production_centroids <- readRDS("centroids_from_2013.RDA")

df <- filter(summary, playoffs==0) %>% 
      dplyr::select(-playoffs, -games, -lift_over_coin_flip, -actual_wins, -predicted_wins) %>%
      mutate(actual = paste0(round(actual_win_rate*100), "%"),
             predicted = paste0(round(pred_win_rate*100), "%"), 
             rank = pred_rank, 
             rank_diff = pred_rank - actual_rank) %>%
      arrange(-pred_win_rate) %>%
      select(-actual_win_rate, -pred_win_rate, -actual_rank, -pred_rank, -accuracy) %>% arrange()
View(df)

grid.newpage()
pdf("summary_season.pdf", height=11*0.4, width=8.5)
mytheme <- gridExtra::ttheme_default(
  core = list(fg_params=list(cex = 0.9)),
  colhead = list(fg_params=list(cex = 0.9)),
  rowhead = list(fg_params=list(cex = 0.9)))
myt1 <- gridExtra::tableGrob(df[1:15, ], rows=NULL, cols=c("Team", "Actual", "Predicted", "Rank", "Rank Diff"), theme = mytheme)
myt2 <- gridExtra::tableGrob(df[16:30, ], rows=NULL, cols=c("Team", "Actual", "Predicted", "Rank", "Rank Diff"), theme = mytheme)
myt <- grid.arrange(myt1, myt2, nrow=1)
grid.draw(myt)
dev.off()


t <- mutate(importances, cluster = as.numeric(substr(Variable, 23, 24))) %>% 
     filter(DATE==as.Date("2016-02-06") & Variable != "home_team_selected") %>%
     select(cluster, Coeff)

surpluses <- left_join(surpluses, t, by="cluster")

m <- max(surpluses$Coeff)
mm <- 0.25

#### Versus Spurs
plot_df1 <- filter(surpluses, selected_team == "San Antonio" & opposing_team=="Golden State" & abs(surplus)>0) 
plot_df1$surplus <- plot_df1$surplus * -1
p1 <- ggplot(data=plot_df1, aes(x=surplus, y=Coeff)) 
p1 <- p1 + scale_x_continuous(limits= c(-mm, mm)) + scale_y_continuous(limits= c(-m, m)) + 
      geom_vline(xintercept=as.numeric(0), linetype=2) +
      geom_hline(yintercept=as.numeric(0), linetype=2) +
      xlab("Archetype Surplus") + ylab("Archetype Importance") +
      theme(axis.text.y=element_blank(), axis.text.x=element_blank()) + 
      ggtitle("San Antonio") + 
  geom_point(size=3, colour="red") + geom_point(data=subset(plot_df1, surplus>0 & Coeff>0), colour="green", size=3) + 
  geom_point(data=subset(plot_df1, surplus<0 & Coeff<0), colour="green", size=3) + 
  geom_point(data=subset(plot_df1, abs(surplus)<0.05 | abs(Coeff)<0.5), colour="yellow", size=3) 
  
p1

#### Versus Cleveland
plot_df2 <- filter(surpluses, selected_team == "Cleveland" & opposing_team=="Golden State" & abs(surplus)>0) 
plot_df2$surplus <- plot_df2$surplus * -1
p2 <- ggplot(data=plot_df2, aes(x=surplus, y=Coeff)) 
p2 <- p2 + scale_x_continuous(limits= c(-mm, mm)) + scale_y_continuous(limits= c(-m, m)) + 
  geom_vline(xintercept=as.numeric(0), linetype=2) +
  geom_hline(yintercept=as.numeric(0), linetype=2) +
  xlab("") + ylab("") + 
  theme(axis.text.y=element_blank(), axis.text.x=element_blank()) + 
  ggtitle("Cleveland") + 
  geom_point(size=3, colour="red") + geom_point(data=subset(plot_df2, surplus>0 & Coeff>0), colour="green", size=3) + 
  geom_point(data=subset(plot_df2, surplus<0 & Coeff<0), colour="green", size=3) +
  geom_point(data=subset(plot_df2, abs(surplus)<0.05 | abs(Coeff)<0.5), colour="yellow", size=3) 
p2

#### Versus OKC
plot_df3 <- filter(surpluses, selected_team == "Oklahoma City" & opposing_team=="Golden State" & abs(surplus)>0) 
plot_df3$surplus <- plot_df3$surplus * -1
p3 <- ggplot(data=plot_df3, aes(x=surplus, y=Coeff)) 
p3 <- p3 + scale_x_continuous(limits= c(-mm, mm)) + scale_y_continuous(limits= c(-m, m)) + 
  geom_vline(xintercept=as.numeric(0), linetype=2) +
  geom_hline(yintercept=as.numeric(0), linetype=2) +
  xlab("Archetype Surplus") + ylab("Archetype Importance") +
  theme(axis.text.y=element_blank(), axis.text.x=element_blank()) + 
  ggtitle("Oklahoma City") + 
  geom_point(size=3, colour="red") + geom_point(data=subset(plot_df3, surplus>0 & Coeff>0), colour="green", size=3) + 
  geom_point(data=subset(plot_df3, surplus<0 & Coeff<0), colour="green", size=3) +
  geom_point(data=subset(plot_df3, abs(surplus)<0.05 | abs(Coeff)<0.5), colour="yellow", size=3)
p3

#### Versus Clips
plot_df4 <- filter(surpluses, selected_team == "LA Clippers" & opposing_team=="Golden State" & abs(surplus)>0 & DATE==as.Date("2015-11-19")) 
plot_df4$surplus <- plot_df4$surplus * -1
p4 <- ggplot(data=plot_df4, aes(x=surplus, y=Coeff)) 
p4 <- p4 + scale_x_continuous(limits= c(-mm, mm)) + scale_y_continuous(limits= c(-m, m)) + 
  geom_vline(xintercept=as.numeric(0), linetype=2) +
  geom_hline(yintercept=as.numeric(0), linetype=2) +
  xlab("") + ylab("") +
  theme(axis.text.y=element_blank(), axis.text.x=element_blank()) + 
  ggtitle("LA Clippers") + 
  geom_point(size=3, colour="red") + geom_point(data=subset(plot_df4, surplus>0 & Coeff>0), colour="green", size=3) + 
  geom_point(data=subset(plot_df4, surplus<0 & Coeff<0), colour="green", size=3) +
  geom_point(data=subset(plot_df4, abs(surplus)<0.05 | abs(Coeff)<0.5), colour="yellow", size=3) 

p4


multiplot(p1, p2, cols=2)

multiplot(p3, p4, cols=2)

stop()












###### Deal with the playoffs
### Summarize team performances -- have to do this twice because the data was collapsed into matchups
playoff1 <- scores %>%
  filter(playoffs==1) %>%
  group_by(selected_team, opposing_team) %>%
  mutate(match=as.numeric(selected_team_win==pred_win)) %>%
  summarise(pred_win=sum(pred_win),
            prob_win_selected_team=sum(prob_win_selected_team),
            win=sum(selected_team_win),
            match=sum(match),
            games=n()) %>%
  rename(team=selected_team) %>%
  ungroup()

playoff2 <- scores %>%
  filter(playoffs==1) %>%
  group_by(opposing_team, selected_team) %>%
  mutate(match = as.numeric((selected_team_win==0)==(pred_win==0))) %>%
  summarise(pred_win=sum(as.numeric(pred_win==0)),
            prob_win_selected_team=sum(1-prob_win_selected_team),
            win=sum(selected_team_win==0),
            match=sum(match),
            games=n()) %>%
  rename(team=opposing_team,
         opposing_team=selected_team) %>%
  ungroup()

playoff_summary <- rbind.data.frame(playoff1, playoff2) %>% group_by(team, opposing_team) %>%
  summarise(win_rate=sum(win), 
            prob_win_selected_team=sum(prob_win_selected_team),
            games=sum(games), 
            pred_win=sum(pred_win), 
            match=sum(match)) %>%
  mutate(prob_win_selected_team = prob_win_selected_team/games, 
         accuracy = match / games, 
         actual_wins=win_rate, 
         predicted_wins=pred_win) %>%
  arrange(-actual_wins) %>%
  select(-match, -prob_win_selected_team, -win_rate, -pred_win, -accuracy, -actual_wins) %>%
  filter(team=="Golden State") %>%
  ungroup()

grid.newpage()
pdf("summary_playoffs.pdf", height=1.5, width=5.5)
mytheme <- gridExtra::ttheme_default(
  core = list(fg_params=list(cex = 1)),
  colhead = list(fg_params=list(cex = 1)),
  rowhead = list(fg_params=list(cex = 1)))
myt <- gridExtra::tableGrob(playoff_summary, rows=NULL, cols=c("Team", "Opposing Team", "Games Played", "Predicted Wins"), theme = mytheme)
grid.draw(myt)
dev.off()


