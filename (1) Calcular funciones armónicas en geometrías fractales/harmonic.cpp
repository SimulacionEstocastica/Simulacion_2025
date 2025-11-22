#include<bits/stdc++.h>
#include <fstream>
using namespace std;

typedef complex<float> cf;
typedef complex<double> cd;
#define all(x) x.begin(),x.end()


// hacer cosas random bien hechas
mt19937_64 MT64;
void pre_init() {
	MT64 = mt19937_64(chrono::system_clock::now().
	time_since_epoch().count());
}

vector<vector<cd>> readComplex(string fileName, int n){
	vector<vector<cd>> res(n,vector<cd>(n,0));
	ifstream lectura(fileName);
	string line; 
	for(int i = 0; i < n; i++){
		int j = 0;
		getline (lectura, line);
		stringstream ss(line);
		double a, b;
		char c;
		while(ss >> a){
			b = 0;
			c = ss.peek();
			if(c == '+'){
				ss >> c;
				ss >> b;
				ss >> c;
			}
			else if(c == '-'){
				ss >> c;
				ss >> b;
				b *= -1;
				ss >> c;
			}
			else if(c == 'i'){
				b = a;
				a = 0;
				ss >> c;
			}
			res[i][j] = cd(a,b);
			c = ss.peek();
			if(ss.peek() == ios::failbit) continue;
			else ss >> c;
			j++;
		}
	}
	lectura.close(); 
	return res;
}


vector<vector<int>> readBoundary(string fileName, int n){
	vector<vector<int>> res(n,vector<int>(n,0));
	ifstream lectura(fileName);
	string line; 
	for(int i = 0; i < n; i++){
		int j = 0;
		getline (lectura, line);
		stringstream ss(line);
		string t;
		while(getline(ss, t, ',')){
			res[i][j] = stoi(t);
			j++;	
		}
	}
	lectura.close(); 
	return res;
}

int getSize(string s){
	int ans = 0;
	stringstream ss(s);
	string t;
	while(getline(ss, t, ',')){
		ans++;
	}
	return ans;
}

int main(){
	//ios::sync_with_stdio(0); cin.tie(0);
	pre_init();
	cout << "Verbose? (true/false)";
	bool verbose; cin >> boolalpha >> verbose; 
	cout << endl;

	string fileName1, fileName2;
	if(verbose) cout << "Ingrese nombre del archivo del campo (valores).\n";
	cin >> fileName1;
	if(verbose) cout << "Ingrese nombre del archivo de la frontera.\n";
	cin >> fileName2;
	auto start = chrono::high_resolution_clock::now();
	ifstream lectura(fileName1);
	string line; getline (lectura, line);
	lectura.close(); 
	int N = getSize(line);
	int center = N/2+1;
	vector<vector<int>> frontera = readBoundary(fileName2, N); 
	vector<vector<cd>> campo = readComplex(fileName1, N); 
	vector<vector<cd>> resultado(N, vector<cd>(N, 0));
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> deltaT = end - start;
	double segundos = deltaT.count();
	string unit = " segundos";
	if(segundos > 60){
		segundos /= 60;
		unit = " minutos";
	}
	if(segundos > 60){
		segundos /= 60;
		unit = " horas";
	}
	cout << "Campos leídos en " << segundos << unit << endl;
	if(verbose) cout << "Ingrese cuantas iteraciones desea hacer: ";
	long long maxn; cin >> maxn;
	if(verbose) cout << "\n";
	static const int dx[4] = {0,1,0,-1};
	static const int dy[4] = {1,0,-1,0};
	cd sum;
	start = chrono::high_resolution_clock::now();
	for(int x = 0; x < N; x++){
		for(int y = 0; y < N; y++){
			if(frontera[x][y]){
				resultado[x][y] = campo[x][y];
				continue;
			}
			sum = {0,0};
			for(int i = 0; i < maxn; i++){
				int a = x,b = y; //coordenada del random walk
				while(!frontera[a][b]){
					int rnd = abs((int)MT64()) & 3;
					int na = a + dx[rnd], nb = b + dy[rnd];
					if(na >= 0 && na < N && nb >= 0 && nb < N){
						a = na;
						b = nb;
					}
				}
				sum += campo[a][b];
			}
			resultado[x][y] = {sum.real()/maxn,sum.imag()/maxn};
		}
		if((x) % (N/10) == 0) cout << ((x)*100)/N + 1 << "% del progreso listo\n";
	}
	end = chrono::high_resolution_clock::now();
	deltaT = end - start;
	// guardar resultado
	if(verbose) cout << "Ingrese nombre archivo de salida: ";
	string outFile; cin >> outFile;
	if(verbose) cout << "\n";	
	ofstream output(outFile);
	for(int x = 0; x < N; x++){
		for(int y = 0; y < N; y++){
			sum = resultado[x][y];
			if(sum.real() != 0) output << sum.real();
			if(sum.imag() != 0){
				if(sum.imag() > 0 && sum.real() != 0) output << "+";
				output << sum.imag();
				output << "i";
			}
			if(sum.real() == 0 && sum.imag() == 0) output << 0;
			if(y != N-1) output << ",";	
		}
		output << endl;
	}
	output.close();
	segundos = deltaT.count();
	unit = " segundos";
	if(segundos > 60){
		segundos /= 60;
		unit = " minutos";
	}
	if(segundos > 60){
		segundos /= 60;
		unit = " horas";
	}
	cout << "Calculado en " << segundos << unit << endl;
	return 0;
}