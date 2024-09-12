#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include "imgcodecs.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>
using namespace std;

struct Point
{
    int x, y;
    Point() : x(0), y(0) {}
    Point(int x, int y) : x(x), y(y) {}

    bool operator==(const Point& other) const
    {
        return x == other.x && y == other.y;
    }

    bool operator<(const Point& other) const
    {
        if (x == other.x)
            return y < other.y;
        return x < other.x;
    }
};

namespace std {
    template <>
    struct hash<Point> {
        size_t operator()(const Point& p) const {
            return hash<int>()(p.x) ^ (hash<int>()(p.y) << 1);
        }
    };
}


class Grid
{
private:
    vector<vector<int>> binaryMatrix;

public:
    Grid(cv::Mat &image)
    {
        preprocessImage(image);
    }
    void preprocessImage(cv::Mat &image)
    {
        cv::threshold(image, image, 128, 1, cv::THRESH_BINARY_INV);
        binaryMatrix.resize(image.rows, vector<int>(image.cols, 0));
        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                binaryMatrix[i][j] = image.at<uchar>(i, j);
            }
        }
    }
    vector<vector<int>> getMatrix()
    {
        return binaryMatrix;
    }
};

class DFS
{
private:
    vector<Point> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // considering only 4 directions of travelling
public:
    vector<Point> findShortestPath(vector<vector<int>> &grid, Point start, Point end)
    {
        int rows = grid.size();
        int cols = grid[0].size();
        stack<Point> stack;
        stack.push(start);
        unordered_map<int, Point> cameFrom;
        vector<vector<bool>> visited(rows, vector<bool>(cols, false));
        visited[start.x][start.y] = true;

        while (!stack.empty())
        {
            Point current = stack.top();
            stack.pop();
            if (current.x == end.x && current.y == end.y)
            {
                vector<Point> path;
                while (!(current.x == start.x && current.y == start.y))
                {
                    path.push_back(current);
                    current = cameFrom[current.x * cols + current.y];
                }
                path.push_back(start);
                reverse(path.begin(), path.end());
                return path;
            }
            for (Point dir : directions)
            {
                Point neighbor(current.x + dir.x, current.y + dir.y);
                if (neighbor.x >= 0 && neighbor.x < rows && neighbor.y >= 0 && neighbor.y < cols && grid[neighbor.x][neighbor.y] == 0 && !visited[neighbor.x][neighbor.y])
                {
                    stack.push(neighbor);
                    visited[neighbor.x][neighbor.y] = true;
                    cameFrom[neighbor.x * cols + neighbor.y] = current;
                }
            }
        }

        return {};
    }
};

class Dijkstra
{
private:
    vector<Point> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

public:
    vector<Point> findShortestPath(vector<vector<int>> &grid, Point start, Point end)
    {
        int rows = grid.size();
        int cols = grid[0].size();
        priority_queue<pair<int, Point>, vector<pair<int, Point>>, greater<pair<int, Point>>> pq;
        pq.push({0, start});

        unordered_map<int, Point> cameFrom;
        vector<vector<int>> dist(rows, vector<int>(cols, INT_MAX));
        dist[start.x][start.y] = 0;

        while (!pq.empty())
        {
            Point current = pq.top().second;
            pq.pop();
            if (current.x == end.x && current.y == end.y)
            {
                vector<Point> path;
                while (!(current.x == start.x && current.y == start.y))
                {
                    path.push_back(current);
                    current = cameFrom[current.x * cols + current.y];
                }
                path.push_back(start);
                reverse(path.begin(), path.end());
                return path;
            }
            for (Point dir : directions)
            {
                Point neighbor(current.x + dir.x, current.y + dir.y);
                if (neighbor.x >= 0 && neighbor.x < rows && neighbor.y >= 0 && neighbor.y < cols && grid[neighbor.x][neighbor.y] == 0)
                {
                    int tentative_dist = dist[current.x][current.y] + 1;

                    if (tentative_dist < dist[neighbor.x][neighbor.y])
                    {
                        cameFrom[neighbor.x * cols + neighbor.y] = current;
                        dist[neighbor.x][neighbor.y] = tentative_dist;
                        pq.push({dist[neighbor.x][neighbor.y], neighbor});
                    }
                }
            }
        }

        return {};
    }
};

class AStar
{
private:
    vector<Point> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int manhattanDistance(Point a, Point b)
    {
        return abs(a.x - b.x) + abs(a.y - b.y);
    }

public:
    vector<Point> findShortestPath(vector<vector<int>> &grid, Point start, Point end)
    {
        int rows = grid.size();
        int cols = grid[0].size();
        priority_queue<pair<int, Point>, vector<pair<int, Point>>, greater<pair<int, Point>>> pq;
        pq.push({0, start});
        unordered_map<int, Point> cameFrom;
        vector<vector<int>> gCost(rows, vector<int>(cols, INT_MAX));
        gCost[start.x][start.y] = 0;
        while (!pq.empty())
        {
            Point current = pq.top().second;
            pq.pop();
            if (current.x == end.x && current.y == end.y)
            {
                vector<Point> path;
                while (!(current.x == start.x && current.y == start.y))
                {
                    path.push_back(current);
                    current = cameFrom[current.x * cols + current.y];
                }
                path.push_back(start);
                reverse(path.begin(), path.end());
                return path;
            }

            for (Point dir : directions)
            {
                Point neighbor(current.x + dir.x, current.y + dir.y);
                if (neighbor.x >= 0 && neighbor.x < rows && neighbor.y >= 0 && neighbor.y < cols && grid[neighbor.x][neighbor.y] == 0)
                {
                    int tentative_gCost = gCost[current.x][current.y] + 1;
                    if (tentative_gCost < gCost[neighbor.x][neighbor.y])
                    {
                        cameFrom[neighbor.x * cols + neighbor.y] = current;
                        gCost[neighbor.x][neighbor.y] = tentative_gCost;
                        int fCost = tentative_gCost + manhattanDistance(neighbor, end);
                        pq.push({fCost, neighbor});
                    }
                }
            }
        }

        return {};
    }
};

int main()
{
    cv::Mat image = cv::imread("maze.png", cv::IMREAD_GRAYSCALE);
    Grid grid(image);
    vector<vector<int>> binaryMatrix = grid.getMatrix();
    Point start(0, 0);
    Point end(binaryMatrix.size() - 1, binaryMatrix[0].size() - 1);
    int choice;
    cout << "Choose Algorithm: 1. DFS 2. Dijkstra 3. A* 4. All\n";
    cin >> choice;
    auto startTime = chrono::high_resolution_clock::now();
    auto endTime = startTime;
    if (choice == 1 || choice == 4)
    {
        DFS dfsSolver;
        startTime = chrono::high_resolution_clock::now();
        vector<Point> dfsPath = dfsSolver.findShortestPath(binaryMatrix, start, end);
        endTime = chrono::high_resolution_clock::now();
        cout << "DFS Execution Time: " << chrono::duration<double>(endTime - startTime).count() << "s\n";

        if (!dfsPath.empty())
        {
            cout << "Path found by DFS:\n";
            for (const auto &p : dfsPath)
            {
                cout << "(" << p.x << ", " << p.y << ") -> ";
            }
            cout << "End\n";
        }
        else
        {
            cout << "No path found by DFS.\n";
        }
    }

    if (choice == 2 || choice == 4)
    {
        Dijkstra dijkstraSolver;
        startTime = chrono::high_resolution_clock::now();
        vector<Point> dijkstraPath = dijkstraSolver.findShortestPath(binaryMatrix, start, end);
        endTime = chrono::high_resolution_clock::now();
        cout << "Dijkstra Execution Time: " << chrono::duration<double>(endTime - startTime).count() << "s\n";

        if (!dijkstraPath.empty())
        {
            cout << "Path found by Dijkstra:\n";
            for (const auto &p : dijkstraPath)
            {
                cout << "(" << p.x << ", " << p.y << ") -> ";
            }
            cout << "End\n";
        }
        else
        {
            cout << "No path found by Dijkstra.\n";
        }
    }

    if (choice == 3 || choice == 4)
    {
        AStar aStarSolver;
        startTime = chrono::high_resolution_clock::now();
        vector<Point> aStarPath = aStarSolver.findShortestPath(binaryMatrix, start, end);
        endTime = chrono::high_resolution_clock::now();
        cout << "A* Execution Time: " << chrono::duration<double>(endTime - startTime).count() << "s\n";

        if (!aStarPath.empty())
        {
            cout << "Path found by A*:\n";
            for (const auto &p : aStarPath)
            {
                cout << "(" << p.x << ", " << p.y << ") -> ";
            }
            cout << "End\n";
        }
        else
        {
            cout << "No path found by A*.\n";
        }
    }

    return 0;
}
