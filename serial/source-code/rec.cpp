#include <vector>
#include <queue>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <functional>

using namespace std;

vector<string> moviesList;

void topRatings(vector<vector<double>> ratingsMat, int user)
{
    priority_queue<pair<double, int>> q;
    for (int i = 0; i < ratingsMat[user].size(); ++i)
    {
        q.push(pair<double, int>(ratingsMat[user][i], i));
    }
    int k = 7; // number of movies to be shown
    cout << "\nTop rated movies by User " << user << endl;
    for (int i = 0; i < k; ++i)
    {
        int ki = q.top().second;
        printf("%s\n", moviesList[ki].c_str());
        q.pop();
    }
}

void makeRec(vector<vector<double>> predict, int user)
{
    priority_queue<pair<double, int>> q;
    for (int i = 0; i < predict[user].size(); ++i)
    {
        q.push(pair<double, int>(predict[user][i], i));
    }
    int k = 7; // number of recomendations to be shown
    cout << "\nRecomendations for User " << user << endl;
    for (int i = 0; i < k; ++i)
    {
        int ki = q.top().second;
        printf("%s\n", moviesList[ki].c_str());

        q.pop();
    }
}

vector<vector<double>> matRead(string file, int row, int col)
{
    ifstream input(file);
    if (!input.is_open())
    {
        cerr << "File is not existing, check the path: \n"
             << file << endl;
        exit(1);
    }
    vector<vector<double>> data(row, vector<double>(col, 0));
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            input >> data[i][j];
        }
    }
    return data;
}

vector<string> movieRead(string file)
{
    vector<string> movies;
    ifstream input(file);
    if (!input.is_open())
    {
        cerr << "File is not existing, check the path: \n"
             << file << endl;
        exit(1);
    }
    string str;
    while (getline(input, str))
    {
        if (str.size() > 0)
            movies.push_back(str);
    }
    return movies;
}

void matWrite(vector<vector<double>> mat, string file)
{
    ofstream output(file);
    int row = mat.size();
    int col = mat[0].size();

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
            output << mat[i][j] << " ";
        output << endl;
    }
}

double norm(vector<double> A) //11993022
{
    double res = 0; //11993022
    for (int i = 0; i < A.size(); ++i) //541105968
    {
        res += pow(A[i], 2); //529112946
    }
    return sqrt(res); //11993022
}

double dotProduct(vector<double> A, vector<double> B) //6084112
{
    double res = 0; //6084112
    for (int i = 0; i < A.size(); ++i) //276637096
    {
        res += A[i] * B[i]; //270552984
    }
    return res; //6084112
}

double adjCosineSimilarity(vector<double> A, vector<double> B) //adjusted cosine similarity (cosine similarity - mean) //5996511
{
    double A_mean = 0; //5996511
    double B_mean = 0; //5996511
    for (int i = 0; i < A.size(); ++i) //270552984
    {
        A_mean += A[i]; //264556473
        B_mean += B[i]; //264556473
    }
    A_mean /= A.size(); //5996511
    B_mean /= B.size(); //5996511
    vector<double> C(A); //5996511
    vector<double> D(B); //5996511
    for (int i = 0; i < A.size(); ++i) //270552984
    {
        C[i] = A[i] - A_mean; //264556473
        D[i] = B[i] - B_mean; //264556473
    }
    return dotProduct(C, D) / (norm(C) * norm(D)); //if output is nan then there is no correlation //11993022
}

void checkCommon(vector<double> A, vector<double> B, vector<double> &C, vector<double> &D) //to check if both A and B have rated //5999329
{
    for (int i = 0; i < A.size(); ++i) //2705697379
    {
        if (A[i] && B[i]) //2699698050
        {
            C.push_back(A[i]); //264556473
            D.push_back(B[i]); //264556473
        }
    }
}

vector<vector<double>> colabFilter(vector<vector<double>> ratingsMat, int usersNum, int itemsNum)
{
    vector<vector<double>> predict(usersNum, vector<double>(itemsNum, 0));
    for (int i = 0; i < usersNum; i++) //Make predictions for each user //269
    {
        for (int j = 0; j < itemsNum; j++) //120868 //Find item j that user i has not scored, and predict user i's score for item j 
        {
            if (ratingsMat[i][j]) //120600 //if movie has already been rated by the user
                continue; //32999
            else //If item j has not been rated by user i, find out users who have rated item j
            {
                vector<double> cosSim;
                vector<double> ratingsOld;
                for (int k = 0; k < usersNum; k++) //If user k has rated item j, calculate the cosSimilarity between user k and user i
                {
                    if (ratingsMat[k][j]) //Find user k who has rated item j
                    {
                        vector<double> commonA, commonB; //  Store the scores of the two items that have been jointly rated in two vectors respectively
                        checkCommon(ratingsMat[i], ratingsMat[k], commonA, commonB); //  check if item has been rated by both users
                        if (!commonA.empty()) //If the two have jointly rated items, calculate the cosine similarity
                        {
                            cosSim.push_back(adjCosineSimilarity(commonA, commonB)); //cosine similarity
                            ratingsOld.push_back(ratingsMat[k][j]); //old ratings
                        }
                    }
                }
                double cosSimSum = 0; //dot product of ratingsOld and cosSim
                if (!cosSim.empty())
                {
                    for (int m = 0; m < cosSim.size(); m++)
                    {
                        cosSimSum += cosSim[m]; //5996511
                    }
                    predict[i][j] = dotProduct(cosSim, ratingsOld) / (cosSimSum); //87601
                    cout << "user " << i << " item " << j << " with predicted rating " << predict[i][j] << endl;
                }
            }
        }
    }
    return predict;
}

int main()
{
    string file1("ratings.txt");
    string file2("movies.txt");

    int row = 268;
    int col = 450;
    vector<vector<double>> ratingsMat = matRead(file1, row, col);
    moviesList = movieRead(file2);
    vector<vector<double>> predict = colabFilter(ratingsMat, row, col);
    matWrite(predict, "predict.txt");

    int uid, check;
    do
    {
        cout << "\nEnter User ID:" << endl;
        cin >> uid;
        topRatings(ratingsMat, uid);
        makeRec(predict, uid);
        cout << "\nRecommend for another user? (1 = Yes, 0 = No)" << endl;
        cin >> check;
    } while (check == 1);

    return 0;
}