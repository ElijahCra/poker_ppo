#pragma once

#include <array>
#include <vector>
#include <sys/types.h>

class Utility {

public:

    Utility();

    static int LookupHandValue(int* pCards);

    static int getWinner(int *p0Cards, int *p1Cards);

    static void EnumerateAll7CardHands();

    //static int LookupSingleHands();

    static bool initLookup();

private:
    static int HR[32487834];
    static bool initialized;
};
