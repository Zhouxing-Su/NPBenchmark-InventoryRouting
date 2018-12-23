#include "LKH.h"
#include "Segment.h"

static thread_local int *Stack;

GainType Penalty_PDTSPL()
{
    static thread_local Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType P = 0;
    int Forward = SUCC(Depot)->Id != Depot->Id + DimensionSaved;
    int StackTop;

    if (!Stack)
        assert(Stack = (int *) malloc(Dim * sizeof(int)));
    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > DimensionSaved)
        StartRoute -= DimensionSaved;
    N = StartRoute;
    do {
        CurrentRoute = N;
        StackTop = -1;
        do {
            if (N->Id <= Dim && N != Depot) {
                if (N->Pickup) {
                    if (StackTop == -1 || Stack[StackTop--] != N->Id)
                        P++;
                    if (P > CurrentPenalty ||
                        (P == CurrentPenalty && CurrentGain <= 0)) {
                        StartRoute = CurrentRoute;
                        return CurrentPenalty + (CurrentGain > 0);
                    }
                } else if (N->Delivery)
                    Stack[++StackTop] = N->Delivery;
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
        P += StackTop + 1;
    } while (N != StartRoute);
    return P;
}
