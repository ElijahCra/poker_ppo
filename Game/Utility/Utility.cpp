//
// Created by Elijah Crain on 8/29/23.
//

#include "Utility.hpp"

#include <cassert>
#include <iostream>
#include <cstring>
#include <filesystem>

bool Utility::initialized = false;

Utility::Utility() {
    initLookup();

    /*
    if (Utility::LookupSingleHands()!=4145){
        throw(std::logic_error("hand ranks not loaded properly try changing path"));
    }; */
}

int Utility::HR[32487834];

bool Utility::initLookup() {

    if (Utility::initialized) {
        return Utility::initialized;
    }

    // Load the HandRanks.DAT file and map it into the HR array
    printf("Loading HandRanks.DAT file...\n");
    memset(HR, 0, sizeof(HR));
    std::string handRanksPathUtility = std::string(PROJECT_SOURCE_DIR)+"/Game/Utility/HandRanks.dat";
    std::string handRanksPathBase = std::string(PROJECT_SOURCE_DIR)+"/HandRanks.dat";
    std::cout<< "Searching in: \n"<< handRanksPathUtility <<std::endl;
    std::cout<< handRanksPathBase <<std::endl;
    FILE * fin = fopen(handRanksPathUtility.c_str(), "rb");
    if (fin == nullptr) {
         {
             fin = fopen(handRanksPathBase.c_str(), "rb");
             if (fin ==nullptr)
             {
                throw(std::runtime_error("did not open properly \n"));
             }
         }
    }

    bool succ = fread(HR, sizeof(HR), 1, fin);	// get the HandRank Array
    if (!succ) {
        throw std::logic_error("didnt read file");
    }
    fclose(fin);
    printf("complete.\n\n");
    initialized = true;
    return true;

}

int Utility::LookupHandValue(int* pCards)
{

    int p = Utility::HR[53 + *pCards++];
    p = Utility::HR[p + *pCards++];
    p = Utility::HR[p + *pCards++];
    p = Utility::HR[p + *pCards++];
    p = Utility::HR[p + *pCards++];
    p = Utility::HR[p + *pCards++];
    return Utility::HR[p + *pCards];
}
/*
int Utility::LookupSingleHands() {
    //printf("Looking up individual hands...\n\n");

    // Create a 7-card poker hand (each card gets a value between 1 and 52)
    int cards[] = { 2, 6, 12, 14, 23, 26, 29 };
    int retVal = Utility::LookupHandValue(cards);
    return retVal;
} */


void Utility::EnumerateAll7CardHands()
{
    // Now let's enumerate every possible 7-card poker hand
    int u0, u1, u2, u3, u4, u5;
    int c0, c1, c2, c3, c4, c5, c6;
    int handTypeSum[10];  // Frequency of hand category (flush, 2 pair, etc)
    int count = 0; // total number of hands enumerated
    memset(handTypeSum, 0, sizeof(handTypeSum));  // do init..

    printf("Enumerating and evaluating all 133,784,560 possible 7-card poker hands...\n\n");

    // On your mark, get set, go...
    //DWORD dwTime = GetTickCount();

    for (c0 = 1; c0 < 47; c0++) {
        u0 = HR[53+c0];
        for (c1 = c0+1; c1 < 48; c1++) {
            u1 = HR[u0+c1];
            for (c2 = c1+1; c2 < 49; c2++) {
                u2 = HR[u1+c2];
                for (c3 = c2+1; c3 < 50; c3++) {
                    u3 = HR[u2+c3];
                    for (c4 = c3+1; c4 < 51; c4++) {
                        u4 = HR[u3+c4];
                        for (c5 = c4+1; c5 < 52; c5++) {
                            u5 = HR[u4+c5];
                            for (c6 = c5+1; c6 < 53; c6++) {

                                handTypeSum[HR[u5+c6] >> 12]++;

                                // JMD: The above line of code is equivalent to:
                                //int finalValue = HR[u5+c6];
                                //int handCategory = finalValue >> 12;
                                //handTypeSum[handCategory]++;

                                count++;
                            }
                        }
                    }
                }
            }
        }
    }

    //dwTime = GetTickCount() - dwTime;

    printf("BAD:              %d\n", handTypeSum[0]);
    printf("High Card:        %d\n", handTypeSum[1]);
    printf("One Pair:         %d\n", handTypeSum[2]);
    printf("Two Pair:         %d\n", handTypeSum[3]);
    printf("Trips:            %d\n", handTypeSum[4]);
    printf("Straight:         %d\n", handTypeSum[5]);
    printf("Flush:            %d\n", handTypeSum[6]);
    printf("Full House:       %d\n", handTypeSum[7]);
    printf("Quads:            %d\n", handTypeSum[8]);
    printf("Straight Flush:   %d\n", handTypeSum[9]);

    // Perform sanity checks. make sure numbers are where they should be
    int testCount = 0;
    for (int index = 0; index < 10; index++)
        testCount += handTypeSum[index];
    if (testCount != count || count != 133784560 || handTypeSum[0] != 0)
    {
        printf("\nERROR!\nERROR!\nERROR!");
        return;
    }

    printf("\nEnumerated %d hands.\n", count);
}

int Utility::getWinner(int *p0Cards, int *p1Cards) {
    int hand0Val = Utility::LookupHandValue(p0Cards);
    int hand1Val = Utility::LookupHandValue(p1Cards);

    //return the winner or tie if they are same value
    if (hand0Val == hand1Val) {
        return 3;
    } else if(hand0Val > hand1Val) {
        return 0;
    } else {
        return 1;
    }
}
