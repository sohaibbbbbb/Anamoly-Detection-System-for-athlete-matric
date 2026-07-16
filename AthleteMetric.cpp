#include <iostream>
#include <fstream>
using namespace std;

struct AthleteMetric {
    int athlete_id;
    float bodyweight;
    float squat;
    float bench;
    float deadlift;
    
    AthleteMetric() : athlete_id(0), bodyweight(0), squat(0), bench(0), deadlift(0) {}
    
    AthleteMetric(int id, float bw, float sq, float bp, float dl) {
        athlete_id = id;
        bodyweight = bw;
        squat = sq;
        bench = bp;
        deadlift = dl;
    }
};

struct Node {
    AthleteMetric data;
    Node* next;
    Node(AthleteMetric val) : data(val), next(nullptr) {}
};

class AthleteQueue {
private:
    Node* front;
    Node* rear;
    int count; 

public:
    AthleteQueue() {
        front = nullptr;
        rear = nullptr;
        count = 0;
    }
    ~AthleteQueue() { while (!isEmpty()) dequeue(); }

    void enqueue(AthleteMetric metric) {
        Node* newNode = new Node(metric);
        if (rear == nullptr) {
            front = rear = newNode;
        } else {
            rear->next = newNode;
            rear = newNode;
        }
        count++;
    }

    void dequeue() {
        if (isEmpty()) return;
        Node* temp = front;
        front = front->next;
        if (front == nullptr) rear = nullptr;
        delete temp;
        count--;
    }

    AthleteMetric peek() const { return front->data; }
    bool isEmpty() const { return front == nullptr; }
    int getSize() const { return count; }
};

void bubble_sort_by_id(AthleteMetric arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j].athlete_id > arr[j + 1].athlete_id) {
                AthleteMetric temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}
int binary_search_by_id(AthleteMetric arr[], int size, int target_id) {
    int left = 0;
    int right = size - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid].athlete_id == target_id) {
            return mid; 
        }
        if (arr[mid].athlete_id < target_id) {
            left = mid + 1; 
        } else {
            right = mid - 1; 
        }
    }
    return -1; 
}

int main() {
    AthleteQueue logQueue;
    
    logQueue.enqueue(AthleteMetric(305, 80.0, 180.0, 120.0, 200.0));
    logQueue.enqueue(AthleteMetric(101, 67.5, 142.5, 95.0, 175.0));
    logQueue.enqueue(AthleteMetric(202, 75.0, 160.0, 110.0, 190.0));

    int size = logQueue.getSize();

    AthleteMetric* metricArray = new AthleteMetric[size];

    for (int i = 0; i < size; i++) {
        metricArray[i] = logQueue.peek();
        logQueue.dequeue();
    }

    bubble_sort_by_id(metricArray, size);

    ofstream outFile("raw_athlete_data.csv");

    if (outFile.is_open()) {
        outFile << "athlete_id,bodyweight,squat,bench,deadlift,anomaly_flag\n";
        
        for (int i = 0; i < size; i++) {
            outFile << metricArray[i].athlete_id << ","
                    << metricArray[i].bodyweight << ","
                    << metricArray[i].squat << ","
                    << metricArray[i].bench << ","
                    << metricArray[i].deadlift << ","
                    << 0 << "\n"; 
        }
        
        outFile.close();
        cout << "Successfully exported sorted memory heap to raw_athlete_data.csv" << endl;
    } else {
        cout << "Error: Could not open CSV file for writing." << endl;
    }

    int target = 202;
    int index = binary_search_by_id(metricArray, size, target);

    if (index != -1) {
        cout << "Athlete " << target << " found! Deadlift: " 
             << metricArray[index].deadlift << "kg" << endl;
    } else {
        cout << "Athlete not found." << endl;
    }

    system("python data_preprocessing.py");
    
    delete[] metricArray;

    return 0;
}