library(xlsx)
library(dplyr)
library(readxl)
library(stringi)
library(tidyr)
library(reshape2)
library(data.table)
library(dplyr)

setwd("/Users/kimlarsen/Code/analytics/analysis/nba/data")

## Read the raw data
read_player_data <- function(season, first_labels){
  data <- data.frame(read_excel(paste0(season, "-Player-BoxScore-Dataset.xlsx"), sheet=1))
  meta <- data.frame(read_excel(paste0(season, "-Player-BoxScore-Dataset.xlsx"), sheet=2, col_names = FALSE))
  labels <- c(first_labels, meta$X1)
  attr(data, "variable.labels") <- labels
  n <- gsub("_$", "", gsub("__", "_", gsub(".", "_", names(data), fixed=T)))
  names(data) <- n
  data <- data %>% rename( 
                 points=PTS, 
                 assists=A,
                 offensive_rebounds=OR,
                 defensive_rebounds=DR, 
                 turnovers=TO,
                 threepointers_made=X3P, 
                 steals=ST,
                 minutes=MIN, 
                 threepoint_attempts=X3PA, 
                 fieldgoal_attempts=FGA, 
                 fieldgoals_made=FG,
                 freethrows_made=FT,
                 freethrow_attempts=FTA, 
                 fouls=PF, 
                 blocks=BL)
  return(data)
}

s1 <- read_player_data("NBA-2012-2013", c("SEASON", "DATE", "PLAYER FULL NAME", "POSITION"))
s2 <- read_player_data("NBA-2013-2014", c("SEASON", "DATE", "PLAYER FULL NAME", "POSITION"))
s3 <- read_player_data("NBA-2014-2015", c("SEASON", "DATE", "PLAYER FULL NAME", "POSITION"))

## Add some indicators
f <- rbind.data.frame(s1, s2, s3) %>%
     mutate(home_team=as.numeric(VENUE_R_H=='H'), 
            road_team=as.numeric(VENUE_R_H=='R'), 
            playoffs=as.numeric(substr(DATA_SET, 6, 13)=="Playoffs"),
            season=ifelse(playoffs==0, as.numeric(substr(DATA_SET, 1, 4)), as.numeric(substr(DATA_SET, 1, 4))-1), 
            playoff_minutes=playoffs*minutes,
            playoff_points=playoffs*points, 
            row_id=row_number())

f$DATE <- as.Date(f$DATE, format="%m/%d/%Y")

## Create an ID
f$cat <- paste0(f$OWN_TEAM, f$OPP_TEAM)
striHelper <- function(x) stri_c(x[stri_order(x)], collapse = "")
f$game_id <- paste0(f$DATE, vapply(stri_split_boundaries(f$cat, type = "character"), striHelper, ""))
f$cat <- NULL

## Team/game level points
team_pts <- f %>% 
            group_by(game_id, OWN_TEAM, VENUE_R_H, DATE) %>%
            summarise(total_playoff_minutes=sum(minutes*playoffs),
                      total_playoff_points=sum(points*playoffs),
                      total_minutes=sum(minutes), 
                      total_points=sum(points),
                      playoffs=max(playoffs)) %>%
            ungroup()

## Game level points
game_pts <- team_pts %>% 
            group_by(game_id) %>%
            mutate(home_team_points=total_points*(VENUE_R_H=='H'), 
                   road_team_points=total_points*(VENUE_R_H=='R')) %>%
            summarise(max_game_points=max(total_points), 
                      home_team_points=sum(home_team_points), 
                      road_team_points=sum(road_team_points)) 

## Random indicator to choose the selected team
set.seed(2015)
game_pts$r <- as.numeric(rbinom(nrow(game_pts), 1, 0.5)>0.5)

## Create win indicators at the game/team level            
team_win <- inner_join(team_pts, select(game_pts, game_id, max_game_points, r), by="game_id") %>%
            mutate(win=as.numeric(total_points==max_game_points)) %>%
            select(-max_game_points)

## Create win indicators at the game level
game_win <- team_win %>% group_by(game_id, DATE) %>%
            mutate(selected_team_win=ifelse(r==1, win*(VENUE_R_H=='H'), win*(VENUE_R_H=='R'))) %>%
            summarise(selected_team_win=max(selected_team_win),
                      playoffs=max(playoffs)) %>%
            ungroup()

## Create a game level summary file to be saved
split <- split(team_win, team_win$game_id)
game_scores <- data.frame(rbindlist(lapply(split, function(x) spread(select(x, game_id, VENUE_R_H, OWN_TEAM), VENUE_R_H, OWN_TEAM))), stringsAsFactors = FALSE) %>%
                   inner_join(select(game_pts, -max_game_points), by="game_id") %>%
                   inner_join(game_win, by="game_id") %>%
                   mutate(selected_team=ifelse(r==1, H, R), 
                          opposing_team=ifelse(r==1, R, H)) %>% 
                   select(-r) %>%
                   rename(home_team_name=H, road_team_name=R)
saveRDS(game_scores, "GAME_SCORES.RDA")

## Create the fill box score file
f <- inner_join(f, select(team_win, -DATE, -VENUE_R_H, -r, -playoffs), by=c("game_id", "OWN_TEAM")) %>%
     inner_join(select(game_scores, -DATE, -playoffs), by="game_id") %>%
     mutate(share_of_minutes=minutes/total_minutes, 
            share_of_playoff_minutes=ifelse(total_playoff_minutes>0, playoff_minutes/total_playoff_minutes, 0),
            share_of_playoff_points=ifelse(total_playoff_points>0, playoff_points/total_playoff_points, 0),
            share_of_points=points/total_points,
            home_points=home_team*points, 
            road_points=road_team*points,
            share_of_minutes_signed = ifelse(OWN_TEAM==selected_team, share_of_minutes, -share_of_minutes),
            home_team_selected = as.numeric(home_team_name==selected_team)) %>%
     dplyr::select(-VENUE_R_H, -TOT)

saveRDS(f, "BOX_SCORES.RDA")