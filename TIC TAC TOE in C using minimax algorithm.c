#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

int completed = 0;   
char arr[3][3];
int platu = 0;

void* iniboard();
void* printboard();
int checkspace(int rop, int cop);
int checkspace1(int rop, int cop);
int checkfilled();
void* playthrd();
void* compthred();
int checkwin(char ch);
int minimax(char board[3][3], int depth, int isMax);
int evaluate(char board[3][3]);
int isMovesLeft(char board[3][3]);

void* iniboard() 
{
    for (int i = 0; i < 3; i++)
     {
        for (int j = 0; j < 3; j++)
         {
            arr[i][j] = ' ';
        }
    }
}

void* printboard() 
{
    for (int i = 0; i < 3; i++) 
    {
        for (int j = 0; j < 3; j++) 
        {
            printf("%c | ", arr[i][j]);
        }
        printf("\n");
    }
}

int checkspace(int rop, int cop) 
{
    if (arr[rop][cop] == ' ') 
    {
        return 1;
    }
    printf(" enter nahi kar sakte ch he wahan \n\n");
    return 0;
}

int checkspace1(int rop, int cop) 
{
    rop -= 1;
    cop -= 1;
    if (arr[rop][cop] == ' ') 
    {
        return 1;
    }
    printf(" enter nahi kar sakte ch he wahan \n\n");
    return 0;
}

int checkfilled() 
{
    for (int i = 0; i < 3; i++)
     {
        for (int j = 0; j < 3; j++) 
        {
            if (arr[i][j] == ' ') 
            {
                return 0;
            }
        }
    }
    printf(" board is filled : Game Draw \n\n");
    return 1;
}

void* playthrd() 
{
    int rop, cop;
    printf("Enter row & col: ");
    scanf("%d %d", &rop, &cop);

    while (rop < 1 || rop > 3 || cop < 1 || cop > 3 || !checkspace1(rop, cop))
    {
        printf("Enter to sahi karo : ");
        scanf("%d %d", &rop, &cop);
    }
    rop -= 1;
    cop -= 1;
    arr[rop][cop] = 'X';

    if (checkwin('X'))
    {
        printf("You won\n");
        completed = 1;
    }
}

void* compthred() 
{
    int bestVal = -1000;
    int bestMoveRow = -1;
    int bestMoveCol = -1;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (arr[i][j] == ' ')
            {
                arr[i][j] = 'O';
                int moveVal = minimax(arr, 0, 0);
                arr[i][j] = ' ';

                if (moveVal > bestVal)
                {
                    bestMoveRow = i;
                    bestMoveCol = j;
                    bestVal = moveVal;
                }
            }
        }
    }

    arr[bestMoveRow][bestMoveCol] = 'O';

    if (checkwin('O'))
    {
        printf("You lose: Computer won\n");
        completed = 1;
    }
}

int checkwin(char ch)
{
    for (int i = 0; i < 3; i++)
    {
        if (arr[i][0] == ch && arr[i][1] == ch && arr[i][2] == ch) return 1;
        if (arr[0][i] == ch && arr[1][i] == ch && arr[2][i] == ch) return 1;
    }
    if (arr[0][0] == ch && arr[1][1] == ch && arr[2][2] == ch) return 1;
    if (arr[0][2] == ch && arr[1][1] == ch && arr[2][0] == ch) return 1;
    return 0;
}

int isMovesLeft(char board[3][3])
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (board[i][j] == ' ')
                return 1;
    return 0;
}

int evaluate(char board[3][3])
{
    for (int row = 0; row < 3; row++)
    {
        if (board[row][0] == board[row][1] && board[row][1] == board[row][2])
        {
            if (board[row][0] == 'O')
                return +10;
            else if (board[row][0] == 'X')
                return -10;
        }
    }

    for (int col = 0; col < 3; col++)
    {
        if (board[0][col] == board[1][col] && board[1][col] == board[2][col])
        {
            if (board[0][col] == 'O')
                return +10;
            else if (board[0][col] == 'X')
                return -10;
        }
    }

    if (board[0][0] == board[1][1] && board[1][1] == board[2][2])
    {
        if (board[0][0] == 'O')
            return +10;
        else if (board[0][0] == 'X')
            return -10;
    }

    if (board[0][2] == board[1][1] && board[1][1] == board[2][0])
    {
        if (board[0][2] == 'O')
            return +10;
        else if (board[0][2] == 'X')
            return -10;
    }

    return 0;
}

int minimax(char board[3][3], int depth, int isMax)
{
    int score = evaluate(board);

    if (score == 10)
        return score - depth;

    if (score == -10)
        return score + depth;

    if (!isMovesLeft(board))
        return 0;

    if (isMax)
    {
        int best = -1000;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (board[i][j] == ' ')
                {
                    board[i][j] = 'O';
                    best = (best > minimax(board, depth + 1, !isMax)) ? best : minimax(board, depth + 1, !isMax);
                    board[i][j] = ' ';
                }
            }
        }
        return best;
    }
    else
    {
        int best = 1000;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (board[i][j] == ' ')
                {
                    board[i][j] = 'X';
                    best = (best < minimax(board, depth + 1, !isMax)) ? best : minimax(board, depth + 1, !isMax);
                    board[i][j] = ' ';
                }
            }
        }
        return best;
    }
}

int main() 
{
    srand(time(NULL));
    pthread_t player, computer;

    iniboard();

    while (!completed && !checkfilled())
    {
        if (platu)
        {
            printboard();
            pthread_create(&player, NULL, playthrd, NULL);
            pthread_join(player, NULL);
            platu = 0;
        }
        else
        {
            pthread_create(&computer, NULL, compthred, NULL);
            pthread_join(computer, NULL);
            platu = 1;
        }
    }

    printboard();

    return 0;
}
