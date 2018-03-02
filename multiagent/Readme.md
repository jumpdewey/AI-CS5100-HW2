###Q1 and Q5
I use almost the same implementation for both Q1 and Q5. 
Firstly, I choose different weight for Ghost, Capsule and food. 
Secondly, traverse all the ghosts and evaulate its effect. For each ghost, if it's right at the same position with pacman, then check whether this ghost is scared or not and plus and minus WEIGHT_GHOST, respectively. If the ghost is not at the pacman's postion, then, we minus the current score with "WEIGHT_GHOST/d" which means if the distance is greater the score will be higher. 
Thirdly, evaluate food's effect by only checking the nearest food through manhattan distance. Due to the differece of "WEIGHT_FOOD/min(distanceToFood)", the score would be higher for those closer food. 
Finally, calculate the capsule's effect if the capsule is eating by pacman, then add WEIGHT_CAPSULE to the current score, otherwise, plus "WEIGHT_FOOD/d" since I want capsule has the same weight as food at this time, because I don't want pacman died on the way chasing for more weighted capsule.

##Time spent on this homework
I spent ~3hrs on q1, ~4hrs on q2,3,4 and ~1hr on q5. 